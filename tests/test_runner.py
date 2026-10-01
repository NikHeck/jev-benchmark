from dataclasses import asdict
from types import SimpleNamespace

import pytest

import benchmark
from benchmark import (
    Category,
    ClassificationError,
    ClassificationResult,
    DeepSeekClassifier,
    JevClassifier,
    OpenAIClassifier,
    Usage,
    run_benchmark,
)


def categories() -> list[Category]:
    return [
        Category(0, "01", "Food", None, 1, (1, 2), False),
        Category(1, "01.1", "Bread", 0, 2, (), True),
        Category(2, "01.2", "Beverages", 0, 2, (3,), False),
        Category(3, "01.2.1", "Coffee", 2, 3, (), True),
    ]


def stub_classifier(classifier_type, decisions, strategy="recursive"):
    """Use the real strategies with local decisions instead of SDK clients."""
    classifier = object.__new__(classifier_type)
    classifier.strategy = strategy
    classifier._init_tree(categories())
    classifier.flat_prompt = benchmark.build_flat_prompt(classifier.categories)
    remaining = iter(decisions)
    calls = []

    def choose(title, candidates, current):
        calls.append((title, [c.id for c in candidates], current.id if current else None))
        decision = next(remaining)
        if isinstance(decision, Exception):
            raise decision
        return decision, Usage(input_tokens=10, output_tokens=2)

    if classifier_type is JevClassifier:
        classifier._choice = choose
    else:
        def request(title, candidates, prompt, recursive):
            current = candidates[0] if recursive and prompt.startswith("CURRENT") else None
            if recursive:
                expected_prompt = benchmark.build_option_prompt(candidates)
                if current:
                    expected_prompt = (
                        f"CURRENT COICOP CATEGORY: {current.id} | {current.code} | {current.title}\n"
                        + expected_prompt
                    )
                assert prompt == expected_prompt
            else:
                assert prompt == classifier.flat_prompt
            return choose(title, candidates, current)

        classifier._request = request
    return classifier, calls


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier])
@pytest.mark.parametrize("decisions, expected, options", [
    ([0, 2, 3], 3, [([0], None), ([0, 1, 2], 0), ([2, 3], 2)]),
    ([0, 2, 2], 2, [([0], None), ([0, 1, 2], 0), ([2, 3], 2)]),
    ([0, 1], 1, [([0], None), ([0, 1, 2], 0)]),
])
def test_recursive_strategies_share_options_and_stopping(classifier_type, decisions, expected, options):
    classifier, calls = stub_classifier(classifier_type, decisions)
    result = classifier.classify("item")
    assert result.category_id == expected
    assert [(ids, current) for _, ids, current in calls] == options
    assert result.usage.requests == len(decisions)
    assert result.usage.input_tokens == 10 * len(decisions)


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier])
@pytest.mark.parametrize("cost_complete", [False, True])
def test_recursive_failure_retains_previous_and_failed_request_usage(classifier_type, cost_complete):
    error = ClassificationError("bad response", usage=Usage(input_tokens=7), cost_complete=cost_complete)
    classifier, _ = stub_classifier(classifier_type, [0, error])
    with pytest.raises(ClassificationError) as captured:
        classifier.classify("item")
    assert str(captured.value) == "bad response"
    assert captured.value.cost_complete is cost_complete
    assert captured.value.usage.input_tokens == 17
    assert captured.value.usage.requests == 2


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier])
def test_recursive_transport_failure_marks_partial_cost_unknown(classifier_type):
    classifier, _ = stub_classifier(classifier_type, [0, RuntimeError("timeout")])
    with pytest.raises(ClassificationError, match="partial billed usage") as captured:
        classifier.classify("item")
    assert captured.value.cost_complete is False
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier])
def test_first_transport_failure_is_not_wrapped(classifier_type):
    error = RuntimeError("timeout")
    classifier, _ = stub_classifier(classifier_type, [error])
    with pytest.raises(RuntimeError) as captured:
        classifier.classify("item")
    assert captured.value is error


def test_jev_local_validation_failure_has_complete_partial_cost():
    classifier, _ = stub_classifier(JevClassifier, [0, ValueError("too many options")])
    with pytest.raises(ClassificationError, match="local validation") as captured:
        classifier.classify("item")
    assert captured.value.cost_complete is True
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier])
def test_direct_strategy_uses_all_categories_in_one_request(classifier_type):
    classifier, calls = stub_classifier(classifier_type, [3], strategy="direct")
    result = classifier.classify("item")
    assert result.category_id == 3
    assert calls == [("item", [0, 1, 2, 3], None)]
    assert result.usage.requests == 1


def test_runner_preserves_order_exact_scoring_and_prediction_log(monkeypatch):
    times = iter(range(100))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    calls = []

    def make_classifier(key, decisions):
        remaining = iter(decisions)

        def classify(title):
            calls.append((key, title))
            decision = next(remaining)
            if isinstance(decision, Exception):
                raise decision
            return ClassificationResult(decision, Usage(input_tokens=10))

        provider, strategy = key.split(".")
        return SimpleNamespace(key=key, provider=provider, strategy=strategy, classify=classify)

    classifiers = [
        make_classifier("openai_luna.direct", [1, 2, 1, RuntimeError("timeout")]),
        make_classifier("typesafe_jev.recursive", [0, 3, 1, 3]),
    ]
    tests = [benchmark.TestCase(1, "Bread"), benchmark.TestCase(3, "Coffee")]
    stats, log = run_benchmark(classifiers, tests, categories(), iterations=2)
    assert calls == [(c.key, test.title) for _ in range(2) for test in tests for c in classifiers]
    assert [(r["iteration"], r["sample_index"], r["overall_sample_index"]) for r in log] == [
        (iteration, row, (iteration - 1) * 2 + row)
        for iteration in (1, 2) for row in (1, 2) for _ in classifiers
    ]
    direct = stats["openai_luna.direct"]
    assert (direct.successes, direct.wrong_predictions, direct.api_errors) == (2, 1, 1)
    assert direct.unknown_cost_attempts == 1
    assert asdict(direct.usage)["requests"] == 3
    assert benchmark.stats_common(direct)["success_rate"] == 0.5
    assert log[1]["status"] == "wrong_prediction"  # A parent gives no partial credit.
    error = log[6]
    assert error["predicted"] is None
    assert error["correct"] is False
    assert error["api_requests"] is None
    assert error["error"] == {"type": "RuntimeError", "message": "timeout"}
    assert all(r["elapsed_seconds"] == 1 for r in log)
    assert set(log[0]) == {
        "iteration", "sample_index", "overall_sample_index", "model", "strategy", "classifier",
        "title", "expected", "predicted", "correct", "status", "elapsed_seconds", "api_requests", "error",
    }


def test_classifier_selection_keeps_order_and_deduplicates(monkeypatch):
    monkeypatch.setattr(benchmark, "OpenAIClassifier", lambda cats, strategy, provider: (provider, strategy))
    monkeypatch.setattr(benchmark, "DeepSeekClassifier", lambda cats, strategy: ("deepseek_flash", strategy))
    monkeypatch.setattr(benchmark, "JevClassifier", lambda cats: ("typesafe_jev", "recursive"))
    assert benchmark.create_classifiers(categories(), [
        "typesafe_jev.recursive", "openai_sol.direct", "typesafe_jev.recursive", "deepseek_flash.recursive",
    ]) == [("typesafe_jev", "recursive"), ("openai_sol", "direct"), ("deepseek_flash", "recursive")]
