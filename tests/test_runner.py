from dataclasses import asdict
import json
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
    OpenAIDecisionsClassifier,
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


def stub_classifier(classifier_type, decisions, strategy="recursive", tree=None):
    """Use the real strategies with local decisions instead of SDK clients."""
    classifier = object.__new__(classifier_type)
    classifier.strategy = strategy
    classifier._init_tree(categories() if tree is None else tree)
    classifier.flat_prompt = benchmark.build_flat_prompt(classifier.leaves)
    remaining = iter(decisions)
    calls = []

    def choose(title, candidates, current):
        calls.append((title, [c.id for c in candidates], current.id if current else None))
        decision = next(remaining)
        if isinstance(decision, Exception):
            raise decision
        return decision, Usage(input_tokens=10, output_tokens=2)

    if classifier_type in (JevClassifier, OpenAIDecisionsClassifier):
        classifier._choice = choose
    else:
        def request(title, candidates, prompt, recursive):
            current = classifier.by_id[candidates[0].parent_id] if recursive and prompt.startswith("CURRENT") else None
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


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier, OpenAIDecisionsClassifier])
@pytest.mark.parametrize("decisions, expected, options", [
    ([0, 2, 3], 3, [([0], None), ([1, 2], 0), ([3], 2)]),
    ([0, 1], 1, [([0], None), ([1, 2], 0)]),
])
def test_recursive_strategies_share_child_options_and_finish_at_leaves(classifier_type, decisions, expected, options):
    classifier, calls = stub_classifier(classifier_type, decisions)
    result = classifier.classify("item")
    assert result.category_id == expected
    assert not classifier.by_id[result.category_id].children_ids
    assert [(ids, current) for _, ids, current in calls] == options
    assert result.usage.requests == len(decisions)
    assert result.usage.input_tokens == 10 * len(decisions)


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier, OpenAIDecisionsClassifier])
def test_recursive_strategy_finishes_immediately_at_a_leaf_root(classifier_type):
    classifier, calls = stub_classifier(
        classifier_type, [0], tree=[Category(0, "01", "Food", None, 1, (), True)]
    )
    result = classifier.classify("item")
    assert result.category_id == 0
    assert calls == [("item", [0], None)]
    assert result.usage.requests == 1


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier, OpenAIDecisionsClassifier])
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


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier, OpenAIDecisionsClassifier])
def test_recursive_transport_failure_marks_partial_cost_unknown(classifier_type):
    classifier, _ = stub_classifier(classifier_type, [0, RuntimeError("timeout")])
    with pytest.raises(ClassificationError, match="partial billed usage") as captured:
        classifier.classify("item")
    assert captured.value.cost_complete is False
    assert captured.value.usage.requests == 1


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, JevClassifier, OpenAIDecisionsClassifier])
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


@pytest.mark.parametrize("classifier_type", [OpenAIClassifier, DeepSeekClassifier, OpenAIDecisionsClassifier])
def test_direct_strategy_uses_only_leaves_in_one_request(classifier_type):
    classifier, calls = stub_classifier(classifier_type, [3], strategy="direct")
    result = classifier.classify("item")
    assert result.category_id == 3
    assert calls == [("item", [1, 3], None)]
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
        make_classifier("openai_luna.direct", [1, 1, 1, RuntimeError("timeout")]),
        make_classifier("typesafe_jev.recursive", [3, 3, 1, 3]),
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
    assert direct.to_output()["success_rate"] == 0.5
    assert log[1]["status"] == "wrong_prediction"
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
    monkeypatch.setattr(benchmark, "JevClassifier", lambda cats, strategy: ("typesafe_jev", strategy))
    monkeypatch.setattr(benchmark, "OpenAIDecisionsClassifier", lambda cats, strategy: ("openai_luna_decisions", strategy))
    assert benchmark.create_classifiers(categories(), [
        "typesafe_jev.recursive_subtree", "openai_sol.direct", "typesafe_jev.recursive_subtree",
        "typesafe_jev.recursive", "openai_luna_decisions.direct", "deepseek_flash.recursive",
        "openai_luna_decisions.recursive", "openai_luna_decisions.recursive_subtree",
        "openai_luna_decisions.direct",
    ]) == [
        ("typesafe_jev", "recursive_subtree"), ("openai_sol", "direct"),
        ("typesafe_jev", "recursive"), ("openai_luna_decisions", "direct"), ("deepseek_flash", "recursive"),
        ("openai_luna_decisions", "recursive"), ("openai_luna_decisions", "recursive_subtree"),
    ]


def test_main_rejects_nonleaf_expected_category_before_creating_clients(tmp_path, monkeypatch):
    category_file = tmp_path / "categories.json"
    category_file.write_text(json.dumps([asdict(c) for c in categories()]))
    input_file = tmp_path / "input.csv"
    input_file.write_text("category_id,title\n0,Food\n")
    monkeypatch.setattr(benchmark, "parse_args", lambda: SimpleNamespace(
        number_samples=1, category_file=category_file, input_file=input_file,
        classifiers=("openai_luna.direct",),
    ))
    monkeypatch.setattr(benchmark, "create_classifiers", lambda *args: pytest.fail(
        "Provider clients must not be created for invalid expected labels"
    ))
    with pytest.raises(ValueError, match="non-leaf category_id 0"):
        benchmark.main()


def test_subtree_strategy_is_available_for_native_choice_apis(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark.py", "--number-samples", "1"])
    assert benchmark.parse_args().classifiers == benchmark.CLASSIFIER_CHOICES
    subtree_choices = [key for key in benchmark.CLASSIFIER_CHOICES if key.endswith(".recursive_subtree")]
    assert subtree_choices == ["openai_luna_decisions.recursive_subtree", "typesafe_jev.recursive_subtree"]
    monkeypatch.setattr("sys.argv", [
        "benchmark.py", "--number-samples", "1", "--classifiers", *subtree_choices,
    ])
    assert benchmark.parse_args().classifiers == subtree_choices
    monkeypatch.setattr("sys.argv", [
        "benchmark.py", "--number-samples", "1", "--classifiers", "openai_luna.recursive_subtree",
    ])
    with pytest.raises(SystemExit):
        benchmark.parse_args()
