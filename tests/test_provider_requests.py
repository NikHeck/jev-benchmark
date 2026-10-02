"""Exercise provider adapters through the real SDK without making API calls."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

import benchmark
from benchmark import Category, ClassificationError, DeepSeekClassifier, JevClassifier, OpenAIClassifier


@pytest.fixture
def provider_http(monkeypatch):
    calls = []
    responses = []
    clients = []
    sdk_client = openai.OpenAI

    def handle(request):
        expected_path = "/responses" if request.url.host == "api.deepseek.com" else "/v1/responses"
        assert request.url.path == expected_path
        assert request.extensions["timeout"] == {
            phase: 600.0 for phase in ("connect", "read", "write", "pool")
        }
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    def make_client(**kwargs):
        client = sdk_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(openai, "OpenAI", make_client)
    yield calls, responses
    for client in clients:
        client.close()


def response(text='{"category_id": 1}', *, status="completed", cached=4, refusal=False):
    content = {"type": "refusal", "refusal": "Cannot classify"} if refusal else {
        "type": "output_text", "text": text, "annotations": [],
    }
    return {
        "id": "test-response", "object": "response", "created_at": 0,
        "model": "test-model", "status": status,
        "output": [{"id": "test-message", "type": "message", "role": "assistant",
                    "status": status, "content": [content]}],
        "usage": {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def categories():
    return [
        Category(0, "01", "Food", None, 1, (1, 2), False),
        Category(1, "01.1", "Bread", 0, 2, (), True),
        Category(2, "01.2", "Beverages", 0, 2, (3,), False),
        Category(3, "01.2.1", "Coffee", 2, 3, (), True),
    ]


@pytest.mark.parametrize("strategy, decisions, option_ids", [
    ("direct", [3], [[0, 1, 2, 3]]),
    ("recursive", [0, 2, 3], [[0], [0, 1, 2], [2, 3]]),
    ("recursive", [0, 2, 2], [[0], [0, 1, 2], [2, 3]]),
])
def test_providers_send_identical_prompts_and_constraints(provider_http, strategy, decisions, option_ids):
    calls, responses = provider_http
    classifiers = [
        OpenAIClassifier(categories(), strategy, "openai_luna"),
        OpenAIClassifier(categories(), strategy, "openai_sol"),
        DeepSeekClassifier(categories(), strategy),
    ]
    payloads = []
    for classifier in classifiers:
        responses.extend(response(json.dumps({"category_id": id_})) for id_ in decisions)
        result = classifier.classify("Kaffee")
        assert result.category_id == decisions[-1]
        assert result.usage.requests == len(decisions)
        payloads.append(calls[-len(decisions):])

    for step, ids in enumerate(option_ids):
        requests = []
        for classifier, batch in zip(classifiers, payloads):
            payload = batch[step].copy()
            assert payload.pop("model") == classifier.model
            requests.append(payload)
        assert requests[0] == requests[1] == requests[2]
        payload = requests[0]
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["max_output_tokens"] == 128
        assert payload["store"] is False
        assert benchmark.MULTILINGUAL_NOTE in payload["instructions"]
        assert "TITLE TO CLASSIFY:\nKaffee" in payload["input"]
        assert ("hierarchical step" in payload["instructions"]) == (strategy == "recursive")
        format_ = payload["text"]["format"]
        assert format_["type"] == "json_schema"
        assert format_["strict"] is True
        assert format_["schema"]["properties"]["category_id"] == {"type": "integer", "enum": ids}
        assert format_["schema"]["required"] == ["category_id"]
        assert format_["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier])
@pytest.mark.parametrize("decisions", [[0, 2, 3], [0, 2, 2]])
def test_jev_shares_recursive_wording_with_native_choices(provider_http, monkeypatch, classifier_type, decisions):
    import typesafe_sdk

    calls, responses = provider_http
    responses.extend(response(json.dumps({"category_id": id_})) for id_ in decisions)
    json_result = classifier_type(categories(), "recursive").classify("Kaffee")
    jev_calls = []
    remaining = iter(decisions)

    def system_one(**kwargs):
        jev_calls.append(kwargs)
        return SimpleNamespace(
            answers={"category": SimpleNamespace(choice=f"id_{next(remaining)}")},
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")

    def make_jev_client(**kwargs):
        assert kwargs["timeout"] == 600.0
        assert kwargs["retry"].max_retries == 0
        return SimpleNamespace(system_one=system_one)

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", make_jev_client)
    jev_result = JevClassifier(categories()).classify("Kaffee")
    assert jev_result.category_id == json_result.category_id == decisions[-1]
    assert jev_result.usage.requests == json_result.usage.requests == 3

    by_id = {c.id: c for c in categories()}
    for step, (json_call, jev_call) in enumerate(zip(calls, jev_calls)):
        assert jev_call["state"] == "Kaffee"
        assert jev_call["model"] == benchmark.TYPESAFE_MODEL
        question = jev_call["questions"]["category"]
        assert isinstance(question, typesafe_sdk.Choice)
        classification_wording = json_call["instructions"].split(" Return JSON only,", 1)[0]
        assert question.instructions.split("\n", 1)[0] == classification_wording
        assert benchmark.MULTILINGUAL_NOTE in question.instructions
        assert "selecting it means stop" in question.instructions
        assert "JSON" not in question.instructions
        ids = json_call["text"]["format"]["schema"]["properties"]["category_id"]["enum"]
        assert question.criteria == {
            f"id_{id_}": f"COICOP {by_id[id_].code}: {by_id[id_].title}" for id_ in ids
        }
        if step:
            current_context = json_call["input"].split("\n", 1)[0]
            assert current_context.startswith("CURRENT COICOP CATEGORY:")
            assert question.instructions == classification_wording + "\n" + current_context
        else:
            assert question.instructions == classification_wording


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier])
@pytest.mark.parametrize("cached", [0, 4, 10, None])
def test_responses_usage_preserves_provider_costs(provider_http, classifier_type, cached):
    _, responses = provider_http
    responses.append(response(cached=cached))
    classifier = classifier_type(categories(), "direct")
    usage = classifier.classify("Bread").usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2
    assert usage.requests == 1
    if classifier_type is DeepSeekClassifier:
        assert usage.cache_hit_tokens == (cached or 0)
        assert usage.cache_miss_tokens == 10 - (cached or 0)
        prices = benchmark.DEEPSEEK_PEAK_PRICES
        expected_cost = ((cached or 0) * prices["cache_hit_input"]
                         + (10 - (cached or 0)) * prices["cache_miss_input"]
                         + 2 * prices["output"]) / 1_000_000
        assert benchmark.deepseek_cost(usage, prices) == pytest.approx(expected_cost)
    else:
        assert usage.cached_input_tokens == (cached or 0)
        assert usage.cache_write_tokens == 1


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier])
@pytest.mark.parametrize("bad_response", [
    response("invalid JSON"),
    response("[]"),
    response("{}"),
    response('{"category_id": 99}'),
    response('{"category_id": "1"}'),
    response('{"category_id": 1.5}'),
    response('{"category_id": true}'),
    response('{"category_id": 1, "extra": 2}'),
    response(status="incomplete"),
    response(status="failed"),
    response(refusal=True),
])
def test_unusable_output_retains_billed_usage(provider_http, classifier_type, bad_response):
    _, responses = provider_http
    responses.append(bad_response)
    classifier = classifier_type(categories(), "direct")
    with pytest.raises(ClassificationError, match="unusable classification") as captured:
        classifier.classify("Bread")
    assert captured.value.cost_complete is True
    assert captured.value.usage.input_tokens == 10
    assert captured.value.usage.output_tokens == 2
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier])
def test_recursive_request_rejects_ids_outside_current_options(provider_http, classifier_type):
    _, responses = provider_http
    responses.extend([response('{"category_id": 0}'), response('{"category_id": 3}')])
    classifier = classifier_type(categories(), "recursive")
    with pytest.raises(ClassificationError, match="not one of") as captured:
        classifier.classify("Coffee")
    assert captured.value.cost_complete is True
    assert captured.value.usage.requests == 2
    assert captured.value.usage.input_tokens == 20
