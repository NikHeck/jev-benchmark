"""Exercise provider adapters through the real SDK without making API calls."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import benchmark
from benchmark import Category, ClassificationError


def make_classifier(provider, tree, strategy="recursive"):
    return benchmark.create_classifiers(tree, [f"{provider}.{strategy}"])[0]


@pytest.fixture
def provider_http(monkeypatch):
    calls = []
    responses = []
    clients = []
    sdk_client = openai.OpenAI

    def handle(request):
        payload = json.loads(request.content)
        expected_path = "/responses" if request.url.host == "api.deepseek.com" else (
            "/v1/decisions" if "questions" in payload else "/v1/responses"
        )
        assert request.url.path == expected_path
        assert request.extensions["timeout"] == {
            phase: 600.0 for phase in ("connect", "read", "write", "pool")
        }
        calls.append(payload)
        return httpx.Response(200, json=responses.pop(0))

    def make_client(**kwargs):
        assert kwargs["max_retries"] == 0
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
    ("direct", [3], [[1, 3]]),
    ("recursive", [0, 2, 3], [[0], [1, 2], [3]]),
    ("recursive", [0, 1], [[0], [1, 2]]),
])
def test_providers_send_identical_prompts_and_constraints(provider_http, strategy, decisions, option_ids):
    calls, responses = provider_http
    classifiers = [
        make_classifier("openai_luna", categories(), strategy),
        make_classifier("openai_sol", categories(), strategy),
        make_classifier("deepseek_flash", categories(), strategy),
    ]
    payloads = []
    for classifier in classifiers:
        responses.extend(response(json.dumps({"category_id": id_})) for id_ in decisions)
        result = classifier.classify("Kaffee")
        assert result.category_id == decisions[-1]
        assert not classifier.tree.by_id[result.category_id].children_ids
        assert result.usage.requests == len(decisions)
        payloads.append(calls[-len(decisions):])

    for step, ids in enumerate(option_ids):
        requests = []
        for classifier, batch in zip(classifiers, payloads):
            payload = batch[step].copy()
            assert payload.pop("model") == classifier.adapter.config.model
            assert payload.pop("reasoning") == {"effort": classifier.adapter.config.reasoning_effort}
            requests.append(payload)
        assert requests[0] == requests[1] == requests[2]
        payload = requests[0]
        assert payload["max_output_tokens"] == 128
        assert payload["store"] is False
        assert benchmark.MULTILINGUAL_NOTE in payload["instructions"]
        assert "COICOP leaf category" in payload["instructions"]
        assert "TITLE TO CLASSIFY:\nKaffee" in payload["input"]
        assert ("hierarchical step" in payload["instructions"]) == (strategy == "recursive")
        format_ = payload["text"]["format"]
        assert format_["type"] == "json_schema"
        assert format_["strict"] is True
        assert format_["schema"]["properties"]["category_id"] == {"type": "integer", "enum": ids}
        assert format_["schema"]["required"] == ["category_id"]
        assert format_["schema"]["additionalProperties"] is False


def test_provider_configuration_drives_requests_and_report(provider_http, monkeypatch):
    calls, responses = provider_http
    config = replace(
        benchmark.PROVIDERS["deepseek_flash"], reasoning_effort="low"
    )
    monkeypatch.setitem(benchmark.PROVIDERS, config.key, config)
    responses.append(response())
    make_classifier(config.key, categories(), "direct").classify("item")
    assert calls[0]["reasoning"] == {"effort": "low"}
    output = benchmark.strategy_output(config.key, "direct", benchmark.Stats())
    assert output["reasoning_effort"] == "low"
    assert benchmark.PROVIDERS["openai_luna"].reasoning_effort == "none"


@pytest.mark.parametrize("provider", ["openai_luna", "deepseek_flash"])
@pytest.mark.parametrize("decisions", [[0, 2, 3], [0, 1]])
def test_jev_shares_recursive_wording_with_native_choices(provider_http, monkeypatch, provider, decisions):
    import typesafe_sdk

    calls, responses = provider_http
    responses.extend(response(json.dumps({"category_id": id_})) for id_ in decisions)
    json_result = make_classifier(provider, categories(), "recursive").classify("Kaffee")
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
    jev_result = make_classifier("typesafe_jev", categories()).classify("Kaffee")
    assert jev_result.category_id == json_result.category_id == decisions[-1]
    assert jev_result.usage.requests == json_result.usage.requests == len(decisions)

    by_id = {c.id: c for c in categories()}
    for step, (json_call, jev_call) in enumerate(zip(calls, jev_calls)):
        assert jev_call["state"] == "Kaffee"
        assert jev_call["model"] == benchmark.PROVIDERS["typesafe_jev"].model
        question = jev_call["questions"]["category"]
        assert isinstance(question, typesafe_sdk.Choice)
        classification_wording = json_call["instructions"].split(" Return JSON only,", 1)[0]
        assert question.instructions.split("\n", 1)[0] == classification_wording
        assert benchmark.MULTILINGUAL_NOTE in question.instructions
        assert "COICOP leaf category" in question.instructions
        assert "continue until a category with no children" in question.instructions
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


@pytest.fixture
def jev_client(monkeypatch):
    import typesafe_sdk

    calls = []
    decisions = []

    def system_one(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            answers={"category": SimpleNamespace(choice=f"id_{decisions.pop(0)}")},
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
    monkeypatch.setattr(
        typesafe_sdk, "TypeSafeClient", lambda **kwargs: SimpleNamespace(system_one=system_one)
    )
    return calls, decisions


@pytest.mark.parametrize("strategy", ["recursive", "recursive_subtree"])
@pytest.mark.parametrize("at_root", [False, True])
@pytest.mark.parametrize("provider", ["typesafe_jev", "openai_luna_decisions"])
def test_native_choice_option_limit_preserves_cost_without_sending_an_extra_request(
    provider_http, jev_client, provider, strategy, at_root
):
    jev_calls, decisions = jev_client
    http_calls, responses = provider_http
    calls = jev_calls if provider == "typesafe_jev" else http_calls
    if at_root:
        tree = [
            Category(id_, str(id_ + 1), "Root", None, 1, (), True)
            for id_ in range(256)
        ]
    else:
        tree = [Category(0, "01", "Food", None, 1, tuple(range(1, 257)), False)]
        tree.extend(
            Category(id_, f"01.{id_}", "Child", 0, 2, (), True)
            for id_ in range(1, 257)
        )
        if provider == "typesafe_jev":
            decisions.append(0)
        else:
            responses.append(decision_response("id_0"))

    classifier = make_classifier(provider, tree, strategy)
    with pytest.raises(ClassificationError, match="at most 255 options") as captured:
        classifier.classify("item")
    expected_requests = 0 if at_root else 1
    assert len(calls) == expected_requests
    assert captured.value.cost_complete is True
    assert captured.value.usage.requests == expected_requests
    assert captured.value.usage.input_tokens == 10 * expected_requests
    output_per_request = 2 if provider == "typesafe_jev" else 0
    assert captured.value.usage.output_tokens == output_per_request * expected_requests


@pytest.mark.parametrize("decisions, expected_contexts, option_ids", [
    ([0, 2, 3], [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3], [2, 3]], [[0, 4], [1, 2], [3]]),
    ([0, 1], [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3]], [[0, 4], [1, 2]]),
    ([4, 5], [[0, 1, 2, 3, 4, 5], [4, 5]], [[0, 4], [5]]),
])
def test_jev_subtree_context_narrows_without_changing_choices(
    jev_client, decisions, expected_contexts, option_ids
):
    calls, remaining = jev_client
    remaining.extend(decisions)
    tree = categories() + [
        Category(4, "02", "Alcohol", None, 1, (5,), False),
        # A collapsed pass-through category skips a COICOP code level.
        Category(5, "02.1.1", "Spirits", 4, 3, (), True),
    ]
    classifier = make_classifier("typesafe_jev", tree, "recursive_subtree")
    result = classifier.classify("Kaffee")
    assert classifier.key == "typesafe_jev.recursive_subtree"
    assert result.category_id == decisions[-1]
    assert result.usage.requests == len(decisions)
    assert result.usage.input_tokens == 10 * len(decisions)
    assert result.usage.output_tokens == 2 * len(decisions)

    by_id = {c.id: c for c in tree}
    for step, (call, expected_ids, choices) in enumerate(zip(calls, expected_contexts, option_ids)):
        assert call["state"] == "Kaffee"
        assert call["model"] == benchmark.PROVIDERS["typesafe_jev"].model
        question = call["questions"]["category"]
        assert question.criteria == {
            f"id_{id_}": f"COICOP {by_id[id_].code}: {by_id[id_].title}" for id_ in choices
        }
        assert question.instructions.startswith(benchmark.build_classification_instructions(recursive=True))
        assert "deeper descendants are context only" in question.instructions
        if step:
            assert benchmark.build_current_category_context(by_id[decisions[step - 1]]) in question.instructions
        else:
            assert "CURRENT COICOP CATEGORY:" not in question.instructions
        rows = question.instructions.split("relationships):\n", 1)[1].splitlines()
        assert [int(row.split(" | ", 1)[0]) for row in rows] == expected_ids
        for row, id_ in zip(rows, expected_ids):
            category = by_id[id_]
            assert row.strip() == f"{id_} | {category.code} | {category.title}"
            depth = 0
            while category.parent_id in expected_ids:
                depth += 1
                category = by_id[category.parent_id]
            assert len(row) - len(row.lstrip()) == 2 * depth


@pytest.mark.parametrize("decisions", [[3], [0, 3], [0, 0], [0, 2, 2]])
@pytest.mark.parametrize("strategy", ["recursive", "recursive_subtree"])
def test_jev_rejects_options_outside_direct_children(jev_client, decisions, strategy):
    calls, remaining = jev_client
    remaining.extend(decisions)
    classifier = make_classifier("typesafe_jev", categories(), strategy)
    with pytest.raises(ClassificationError, match="not in the supplied options") as captured:
        classifier.classify("Coffee")
    if strategy == "recursive_subtree":
        assert "3 | 01.2.1 | Coffee" in calls[-1]["questions"]["category"].instructions
    assert captured.value.cost_complete is True
    assert captured.value.usage.requests == len(decisions)
    assert captured.value.usage.input_tokens == 10 * len(decisions)


def test_jev_subtree_context_can_include_more_than_255_categories(jev_client):
    calls, remaining = jev_client
    remaining.extend([0, 1, 3])
    tree = [
        Category(0, "01", "Food", None, 1, (1, 2), False),
        Category(1, "01.1", "Group one", 0, 2, tuple(range(3, 133)), False),
        Category(2, "01.2", "Group two", 0, 2, tuple(range(133, 263)), False),
    ]
    for parent in tree[1:]:
        tree.extend(
            Category(id_, f"{parent.code}.{index}", f"Item {id_}", parent.id, 3, (), True)
            for index, id_ in enumerate(parent.children_ids, start=1)
        )
    result = make_classifier("typesafe_jev", tree, "recursive_subtree").classify("item")
    assert result.category_id == 3
    assert result.usage.requests == 3
    assert [len(call["questions"]["category"].criteria) for call in calls] == [1, 2, 130]
    rows = calls[0]["questions"]["category"].instructions.split("relationships):\n", 1)[1].splitlines()
    assert len(rows) == len(tree) == 263


def decision_response(choice="id_3"):
    return {
        "model": "gpt-6-luna",
        "answers": [{
            "type": "choice", "name": "category", "choice": choice,
            "confidence": 0.9,
            "probabilities": [{"value": choice, "probability": 1.0}],
        }],
        "usage": {
            "input_tokens": 10, "output_tokens": 0, "total_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


@pytest.mark.parametrize("strategy", ["recursive", "recursive_subtree"])
@pytest.mark.parametrize("decisions", [[0, 2, 3], [0, 1], [4, 5]])
def test_decisions_matches_jev_choices_context_and_leaf_results(
    provider_http, jev_client, strategy, decisions
):
    calls, responses = provider_http
    jev_calls, remaining = jev_client
    remaining.extend(decisions)
    responses.extend(decision_response(f"id_{id_}") for id_ in decisions)
    tree = categories() + [
        Category(4, "02", "Alcohol", None, 1, (5,), False),
        Category(5, "02.1.1", "Spirits", 4, 3, (), True),
    ]
    jev_result = make_classifier("typesafe_jev", tree, strategy).classify("Kaffee")
    classifier = make_classifier("openai_luna_decisions", tree, strategy)
    result = classifier.classify("Kaffee")
    assert classifier.key == f"openai_luna_decisions.{strategy}"
    assert result.category_id == jev_result.category_id == decisions[-1]
    assert result.usage.requests == jev_result.usage.requests == len(decisions)
    assert result.usage.input_tokens == 10 * len(decisions)
    assert result.usage.output_tokens == 0
    for call, jev_call in zip(calls, jev_calls):
        assert set(call) == {"model", "input", "questions"}
        assert call["model"] == benchmark.PROVIDERS["openai_luna"].model
        assert call["input"] == jev_call["state"]
        assert len(call["questions"]) == 1
        question = call["questions"][0]
        jev_question = jev_call["questions"]["category"]
        assert question["type"] == "choice"
        assert question["name"] == "category"
        assert question["instructions"] == jev_question.instructions
        assert {c["value"]: c["description"] for c in question["choices"]} == jev_question.criteria


@pytest.mark.parametrize("strategy", ["recursive", "recursive_subtree"])
def test_decisions_recursive_choices_fit_limit_with_full_taxonomy(provider_http, strategy):
    calls, responses = provider_http
    tree_path = Path(__file__).resolve().parents[1] / "category_input.json"
    tree = benchmark.load_categories(tree_path)
    leaves = [c for c in tree if not c.children_ids]
    assert len(leaves) > 255
    by_id = {c.id: c for c in tree}
    leaf = next(c for c in leaves if c.code == "01.1.4.4.5")
    path = [leaf]
    while path[-1].parent_id is not None:
        path.append(by_id[path[-1].parent_id])
    path.reverse()
    responses.extend(decision_response(f"id_{c.id}") for c in path)
    classifier = make_classifier("openai_luna_decisions", tree, strategy)
    result = classifier.classify("Unsweetened oat drink")
    assert result.category_id == leaf.id
    assert result.usage.requests == len(path)
    assert len(calls) == len(path)
    for index, call in enumerate(calls):
        current = None if index == 0 else path[index - 1]
        candidates = classifier.tree.roots if current is None else classifier.tree.children(current)
        question = call["questions"][0]
        assert len(question["choices"]) <= 255
        assert question["choices"] == [
            {"value": f"id_{c.id}", "description": f"COICOP {c.code}: {c.title}"} for c in candidates
        ]
        if strategy == "recursive_subtree":
            assert classifier.tree.subtree_context(current) in question["instructions"]


@pytest.mark.parametrize("answers", [
    [], None, {}, [None], [{"type": "refusal", "name": "category"}],
    [{"type": "predicate", "name": "category", "probability": 0.9}],
    [{"type": "choice", "name": "other", "choice": "id_3"}],
    [{"type": "choice", "name": "category"}],
    [{"type": "choice", "name": "category", "choice": True}],
    [{"type": "choice", "name": "category", "choice": 3}],
    [{"type": "choice", "name": "category", "choice": "id_0"}],
    [{"type": "choice", "name": "category", "choice": "id_2"}],
    [{"type": "choice", "name": "category", "choice": "id_99"}],
    [{"type": "choice", "name": "category", "choice": "id_03"}],
    [{"type": "choice", "name": "category", "choice": "id_3"}] * 2,
])
def test_decisions_unusable_answers_retain_billed_usage(provider_http, answers):
    _, responses = provider_http
    payload = decision_response()
    payload["answers"] = answers
    responses.append(payload)
    leaves = [replace(c, parent_id=None) for c in categories() if c.is_leaf]
    classifier = make_classifier("openai_luna_decisions", leaves)
    with pytest.raises(ClassificationError, match="unusable classification") as captured:
        classifier.classify("Coffee")
    assert captured.value.cost_complete is True
    assert captured.value.usage.input_tokens == 10
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("strategy", ["recursive", "recursive_subtree"])
@pytest.mark.parametrize("decisions", [[0, 3], [0, 0], [0, 2, 2]])
def test_decisions_rejects_options_outside_direct_children(provider_http, strategy, decisions):
    _, responses = provider_http
    responses.extend(decision_response(f"id_{id_}") for id_ in decisions)
    classifier = make_classifier("openai_luna_decisions", categories(), strategy)
    with pytest.raises(ClassificationError, match="not in the supplied options") as captured:
        classifier.classify("Coffee")
    assert captured.value.cost_complete is True
    assert captured.value.usage.requests == len(decisions)
    assert captured.value.usage.input_tokens == 10 * len(decisions)


@pytest.mark.parametrize("usage", [
    None, {}, {"input_tokens": "10"}, {"input_tokens": True}, {"input_tokens": -1},
])
def test_decisions_missing_usage_preserves_previous_cost_as_lower_bound(provider_http, usage):
    _, responses = provider_http
    payload = decision_response("id_1")
    payload["usage"] = usage
    responses.extend([decision_response("id_0"), payload])
    classifier = make_classifier("openai_luna_decisions", categories(), "recursive_subtree")
    with pytest.raises(ClassificationError, match="input usage") as captured:
        classifier.classify("Bread")
    assert captured.value.cost_complete is False
    assert captured.value.usage.requests == 2
    assert captured.value.usage.input_tokens == 10


@pytest.mark.parametrize("usage_details", [
    {"input_tokens_details": "invalid"},
    {"input_tokens_details": {"cached_tokens": "invalid"}},
    {"output_tokens": "invalid"},
])
def test_decisions_invalid_usage_details_retain_input_cost(provider_http, usage_details):
    _, responses = provider_http
    payload = decision_response()
    payload["usage"].update(usage_details)
    responses.append(payload)
    leaves = [replace(c, parent_id=None) for c in categories() if c.is_leaf]
    classifier = make_classifier("openai_luna_decisions", leaves)
    with pytest.raises(ClassificationError) as captured:
        classifier.classify("Coffee")
    assert captured.value.cost_complete is True
    assert captured.value.usage.input_tokens == 10
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("provider", ["openai_luna", "deepseek_flash"])
@pytest.mark.parametrize("cached", [0, 4, 10, None])
def test_responses_usage_preserves_provider_costs(provider_http, provider, cached):
    _, responses = provider_http
    responses.append(response(cached=cached))
    classifier = make_classifier(provider, categories(), "direct")
    usage = classifier.classify("Bread").usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2
    assert usage.requests == 1
    if provider == "deepseek_flash":
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


@pytest.mark.parametrize("provider", ["openai_luna", "deepseek_flash"])
@pytest.mark.parametrize("bad_response", [
    response("invalid JSON"),
    response("[]"),
    response("{}"),
    response('{"category_id": 99}'),
    response('{"category_id": 0}'),
    response('{"category_id": 2}'),
    response('{"category_id": "1"}'),
    response('{"category_id": 1.5}'),
    response('{"category_id": true}'),
    response('{"category_id": 1, "extra": 2}'),
    response(status="incomplete"),
    response(status="failed"),
    response(refusal=True),
])
def test_unusable_output_retains_billed_usage(provider_http, provider, bad_response):
    _, responses = provider_http
    responses.append(bad_response)
    classifier = make_classifier(provider, categories(), "direct")
    with pytest.raises(ClassificationError, match="unusable classification") as captured:
        classifier.classify("Bread")
    assert captured.value.cost_complete is True
    assert captured.value.usage.input_tokens == 10
    assert captured.value.usage.output_tokens == 2
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("provider", ["openai_luna", "deepseek_flash"])
@pytest.mark.parametrize("decisions", [[0, 3], [0, 0], [0, 2, 2]])
def test_recursive_request_rejects_ids_outside_direct_children(provider_http, provider, decisions):
    _, responses = provider_http
    responses.extend(response(json.dumps({"category_id": id_})) for id_ in decisions)
    classifier = make_classifier(provider, categories(), "recursive")
    with pytest.raises(ClassificationError, match="not one of") as captured:
        classifier.classify("Coffee")
    assert captured.value.cost_complete is True
    assert captured.value.usage.requests == len(decisions)
    assert captured.value.usage.input_tokens == 10 * len(decisions)
