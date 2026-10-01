#!/usr/bin/env python3
"""Benchmark direct and recursive COICOP classification across OpenAI, DeepSeek and Jev."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol


OPENAI_MODELS = {
    "openai_luna": os.getenv("OPENAI_LUNA_MODEL", "gpt-6-luna"),
    "openai_sol": os.getenv("OPENAI_SOL_MODEL", "gpt-6-sol"),
}
OPENAI_REASONING_EFFORT = "none"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")

# USD per 1M tokens. Verify before long benchmark runs; provider prices can change.
OPENAI_PRICES = {
    "openai_luna": {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50},
    "openai_sol": {"input": 2.00, "cached_input": 0.20, "cache_write": 2.50, "output": 10.00},
}
STRATEGIES = {
    "openai_luna": ("direct", "recursive"),
    "openai_sol": ("direct", "recursive"),
    "deepseek_flash": ("direct", "recursive"),
    "typesafe_jev": ("recursive",),
}
CLASSIFIER_CHOICES = tuple(f"{provider}.{strategy}" for provider, strategies in STRATEGIES.items() for strategy in strategies)
DEEPSEEK_PEAK_PRICES = {"cache_hit_input": 0.006, "cache_miss_input": 0.30, "output": 1.20}
DEEPSEEK_OFFPEAK_PRICES = {"cache_hit_input": 0.003, "cache_miss_input": 0.15, "output": 0.60}
TYPESAFE_INPUT_PRICE = 0.042

MULTILINGUAL_NOTE = (
    "The item title may be in any language and is not necessarily English. "
    "Interpret the title in its original language before choosing the COICOP category."
)


@dataclass(frozen=True)
class Category:
    id: int
    code: str
    title: str
    parent_id: int | None
    level: int
    children_ids: tuple[int, ...]
    is_leaf: bool


@dataclass(frozen=True)
class TestCase:
    category_id: int
    title: str


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1

    def add(self, other: "Usage") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class ClassificationResult:
    category_id: int
    usage: Usage


class ClassificationError(RuntimeError):
    def __init__(self, message: str, *, usage: Usage | None = None, cost_complete: bool = False) -> None:
        super().__init__(message)
        self.usage = usage
        self.cost_complete = cost_complete


@dataclass
class Stats:
    successes: int = 0
    wrong_predictions: int = 0
    api_errors: int = 0
    known_cost_attempts: int = 0
    unknown_cost_attempts: int = 0
    total_seconds: float = 0.0
    usage: Usage = field(default_factory=lambda: Usage(requests=0))
    complete_cost_usage: Usage = field(default_factory=lambda: Usage(requests=0))
    error_examples: list[str] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return self.wrong_predictions + self.api_errors

    @property
    def samples(self) -> int:
        return self.successes + self.failures

    def _record_usage(self, elapsed: float, usage: Usage) -> None:
        self.known_cost_attempts += 1
        self.total_seconds += elapsed
        self.usage.add(usage)
        self.complete_cost_usage.add(usage)

    def record_prediction(
        self,
        elapsed: float,
        usage: Usage,
        expected: Category,
        predicted: Category,
    ) -> None:
        if expected.id == predicted.id:
            self.successes += 1
        else:
            self.wrong_predictions += 1
        self._record_usage(elapsed, usage)

    def record_error(self, elapsed: float, exc: Exception) -> None:
        self.api_errors += 1
        self.total_seconds += elapsed

        if isinstance(exc, ClassificationError):
            if exc.usage is not None:
                self.usage.add(exc.usage)
            if exc.cost_complete:
                self.known_cost_attempts += 1
                if exc.usage is not None:
                    self.complete_cost_usage.add(exc.usage)
            else:
                self.unknown_cost_attempts += 1
        else:
            self.unknown_cost_attempts += 1

        if len(self.error_examples) < 5:
            self.error_examples.append(f"{type(exc).__name__}: {exc}")


class Classifier(Protocol):
    key: str
    provider: str
    strategy: str

    def classify(self, title: str) -> ClassificationResult: ...


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def build_flat_prompt(categories: list[Category]) -> str:
    rows = ["AVAILABLE COICOP CATEGORIES (id | code | title):"]
    rows.extend(f"{c.id} | {c.code} | {c.title}" for c in categories)
    return "\n".join(rows)


def build_option_prompt(categories: list[Category]) -> str:
    rows = ["AVAILABLE OPTIONS (id | code | title):"]
    rows.extend(f"{c.id} | {c.code} | {c.title}" for c in categories)
    return "\n".join(rows)


def coicop_sort_key(code: str) -> tuple[int, ...]:
    return tuple(int(part) for part in code.split("."))


def load_categories(path: Path) -> list[Category]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("category_input.json must be a non-empty JSON list.")

    # Tree metadata is preferred, but derive it for compatibility with older generated files.
    basic = [
        {"id": int(item["id"]), "code": str(item["code"]), "title": str(item["title"]), **item}
        for item in raw
    ]
    ids = [int(item["id"]) for item in basic]
    if ids != list(range(len(basic))):
        raise ValueError("Category ids must start at 0 and increase by 1 without gaps.")
    if len({str(item["code"]) for item in basic}) != len(basic):
        raise ValueError("COICOP codes must be unique.")

    code_to_id = {str(item["code"]): int(item["id"]) for item in basic}
    derived_children: dict[int, list[int]] = {int(item["id"]): [] for item in basic}
    derived_parent: dict[int, int | None] = {}
    for item in basic:
        cid, code = int(item["id"]), str(item["code"])
        pcode = code.rsplit(".", 1)[0] if "." in code else None
        pid = code_to_id.get(pcode) if pcode is not None else None
        derived_parent[cid] = pid
        if pid is not None:
            derived_children[pid].append(cid)

    categories: list[Category] = []
    for item in basic:
        cid, code = int(item["id"]), str(item["code"])
        parent_id = item.get("parent_id", derived_parent[cid])
        parent_id = int(parent_id) if parent_id is not None else None
        children_raw = item.get("children_ids", derived_children[cid])
        children_ids = tuple(int(v) for v in children_raw)
        level = int(item.get("level", code.count(".") + 1))
        is_leaf = bool(item.get("is_leaf", not children_ids))
        categories.append(Category(cid, code, str(item["title"]), parent_id, level, children_ids, is_leaf))

    by_id = {c.id: c for c in categories}
    for c in categories:
        if c.parent_id is not None and c.parent_id not in by_id:
            raise ValueError(f"Category {c.id} has unknown parent_id {c.parent_id}.")
        for child_id in c.children_ids:
            if child_id not in by_id:
                raise ValueError(f"Category {c.id} has unknown child id {child_id}.")
    return categories


def load_tests(path: Path, valid_ids: set[int]) -> list[TestCase]:
    """Load benchmark cases from CSV.

    Only ``category_id`` and ``title`` are required. Any additional columns are
    deliberately ignored so callers can keep sample IDs, notes, merchant names,
    or other benchmark metadata in the same file without changing the runner.
    """
    tests: list[TestCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                "input CSV must contain a header row with columns: category_id,title."
            )

        # Be forgiving about whitespace/case in the header while requiring the
        # two semantic column names. Extra columns are ignored below.
        normalized = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name is not None and name.strip()
        }
        missing = {"category_id", "title"} - set(normalized)
        if missing:
            raise ValueError(
                "input CSV is missing required column(s): " + ", ".join(sorted(missing))
            )

        category_id_column = normalized["category_id"]
        title_column = normalized["title"]
        for row_number, row in enumerate(reader, start=2):
            raw_category_id = (row.get(category_id_column) or "").strip()
            title = (row.get(title_column) or "").strip()
            if not raw_category_id and not title:
                continue
            if not raw_category_id:
                raise ValueError(f"input CSV row {row_number} has an empty category_id.")
            try:
                category_id = int(raw_category_id)
            except ValueError as exc:
                raise ValueError(
                    f"input CSV row {row_number} has a non-integer category_id: "
                    f"{raw_category_id!r}."
                ) from exc
            if category_id not in valid_ids:
                raise ValueError(
                    f"input CSV row {row_number} references unknown category_id {category_id}."
                )
            if not title:
                raise ValueError(f"input CSV row {row_number} has an empty title.")
            tests.append(TestCase(category_id=category_id, title=title))

    if not tests:
        raise ValueError("input CSV must contain at least one test row.")
    return tests


class TreeMixin:
    def _init_tree(self, categories: list[Category]) -> None:
        self.categories = categories
        self.by_id = {c.id: c for c in categories}
        self.roots = sorted((c for c in categories if c.parent_id is None), key=lambda c: coicop_sort_key(c.code))

    def _next_options(self, current: Category) -> list[Category]:
        # Include the current category as a valid stopping choice, then its children.
        children = [self.by_id[cid] for cid in current.children_ids]
        children.sort(key=lambda c: coicop_sort_key(c.code))
        return [current, *children]

    def _classify_recursive(
        self,
        title: str,
        choose: Callable[[str, list[Category], Category | None], tuple[int, Usage]],
        failure_context: str,
        *,
        local_validation: bool = False,
    ) -> ClassificationResult:
        """Walk the tree and retain billed usage if a later decision fails."""
        total = Usage(requests=0)
        try:
            selected_id, usage = choose(title, self.roots, None)
            total.add(usage)
            selected = self.by_id[selected_id]
            while selected.children_ids:
                next_id, usage = choose(title, self._next_options(selected), selected)
                total.add(usage)
                if next_id == selected.id:
                    break
                selected = self.by_id[next_id]
            return ClassificationResult(selected.id, total)
        except ClassificationError as exc:
            if exc.usage is not None:
                total.add(exc.usage)
            raise ClassificationError(str(exc), usage=total, cost_complete=exc.cost_complete) from exc
        except Exception as exc:
            if local_validation and isinstance(exc, ValueError):
                raise ClassificationError(
                    f"{failure_context} failed local validation: {exc}", usage=total, cost_complete=True
                ) from exc
            if total.requests > 0:
                raise ClassificationError(
                    f"{failure_context} failed after partial billed usage: {exc}",
                    usage=total,
                    cost_complete=False,
                ) from exc
            raise


class JsonClassifier(TreeMixin):
    """Common direct and recursive strategies for the JSON-based providers."""

    strategy: str
    failure_context: str
    flat_prompt: str

    def _request(self, title: str, candidates: list[Category], prompt: str, recursive: bool) -> tuple[int, Usage]:
        raise NotImplementedError

    def _choose_recursive(
        self, title: str, candidates: list[Category], current: Category | None
    ) -> tuple[int, Usage]:
        prompt = build_option_prompt(candidates)
        if current is not None:
            prompt = f"CURRENT COICOP CATEGORY: {current.id} | {current.code} | {current.title}\n" + prompt
        return self._request(title, candidates, prompt, recursive=True)

    def classify(self, title: str) -> ClassificationResult:
        if self.strategy == "direct":
            category_id, usage = self._request(title, self.categories, self.flat_prompt, recursive=False)
            return ClassificationResult(category_id, usage)
        return self._classify_recursive(title, self._choose_recursive, self.failure_context)


class OpenAIClassifier(JsonClassifier):
    failure_context = "OpenAI recursive classification"

    def __init__(self, categories: list[Category], strategy: str, provider: str = "openai_luna") -> None:
        self.provider = provider
        self.strategy = strategy
        self.key = f"{self.provider}.{strategy}"
        self._init_tree(categories)
        from openai import OpenAI

        self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"), max_retries=0)
        self.flat_prompt = build_flat_prompt(categories)

    def _request(self, title: str, candidates: list[Category], prompt: str, recursive: bool) -> tuple[int, Usage]:
        valid_ids = [c.id for c in candidates]
        instructions = (
            "Classify the expense or product title into exactly one COICOP category. "
            "Use only a category id from the supplied list. "
            f"{MULTILINGUAL_NOTE}"
        )
        if recursive:
            instructions += " In this hierarchical step, selecting the current category means stop; otherwise choose its best child."
        response = self.client.responses.create(
            model=OPENAI_MODELS[self.provider],
            reasoning={"effort": OPENAI_REASONING_EFFORT},
            instructions=instructions,
            input=f"{prompt}\n\nTITLE TO CLASSIFY:\n{title}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "coicop_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"category_id": {"type": "integer", "enum": valid_ids}},
                        "required": ["category_id"],
                        "additionalProperties": False,
                    },
                }
            },
            store=False,
        )
        u = response.usage
        details = getattr(u, "input_tokens_details", None)
        usage = Usage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            cache_write_tokens=int(getattr(details, "cache_write_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        )
        try:
            category_id = int(json.loads(response.output_text)["category_id"])
            if category_id not in valid_ids:
                raise ValueError(f"category_id {category_id} is not one of {valid_ids}")
        except Exception as exc:
            raise ClassificationError(
                f"OpenAI returned an unusable classification: {exc}", usage=usage, cost_complete=True
            ) from exc
        return category_id, usage


class DeepSeekClassifier(JsonClassifier):
    provider = "deepseek_flash"
    failure_context = "DeepSeek recursive classification"

    def __init__(self, categories: list[Category], strategy: str) -> None:
        self.strategy = strategy
        self.key = f"{self.provider}.{strategy}"
        self._init_tree(categories)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=require_env("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", max_retries=0
        )
        self.flat_prompt = build_flat_prompt(categories)

    def _request(self, title: str, candidates: list[Category], prompt: str, recursive: bool) -> tuple[int, Usage]:
        valid_ids = {c.id for c in candidates}
        system = (
            "Classify the expense or product title into exactly one COICOP category. "
            "Return JSON only, exactly like {\"category_id\": 123}. "
            "Use only an id from the supplied category list. "
            f"{MULTILINGUAL_NOTE}"
        )
        if recursive:
            system += " In this hierarchical step, selecting the current category means stop; otherwise choose its best child."
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{prompt}\n\nTITLE TO CLASSIFY:\n{title}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=32,
            reasoning_effort="none",
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content or ""
        u = response.usage
        prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
        hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
        miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
        if hit == 0 and miss == 0:
            miss = prompt_tokens
        usage = Usage(
            input_tokens=prompt_tokens,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            output_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )
        try:
            category_id = int(json.loads(content)["category_id"])
            if category_id not in valid_ids:
                raise ValueError(f"category_id {category_id} is not in the supplied options")
        except Exception as exc:
            raise ClassificationError(
                f"DeepSeek returned an unusable classification: {exc}", usage=usage, cost_complete=True
            ) from exc
        return category_id, usage


class JevClassifier(TreeMixin):
    provider = "typesafe_jev"
    strategy = "recursive"
    key = "typesafe_jev.recursive"

    def __init__(self, categories: list[Category]) -> None:
        require_env("TYPESAFE_API_KEY")
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

        self._init_tree(categories)
        self._choice_type = Choice
        self.client = TypeSafeClient(retry=RetryPolicy(max_retries=0))

    def _choice(self, title: str, candidates: list[Category], current: Category | None) -> tuple[int, Usage]:
        if len(candidates) > 255:
            raise ValueError(f"Jev Choice supports at most 255 options; this node has {len(candidates)}.")
        criteria = {f"id_{c.id}": f"COICOP {c.code}: {c.title}" for c in candidates}
        if current is None:
            instructions = "Choose the top-level COICOP category that best matches this expense or product title. "
        else:
            instructions = (
                f"The current category is COICOP {current.code}: {current.title}. "
                "Choose the current category itself if it is the best final classification; otherwise choose the best child. "
            )
        instructions += MULTILINGUAL_NOTE
        response = self.client.system_one(
            model=TYPESAFE_MODEL,
            state=title,
            questions={"category": self._choice_type(instructions=instructions, criteria=criteria)},
        )
        u = response.usage
        usage = Usage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        )
        try:
            key = str(response.answers["category"].choice)
            if not key.startswith("id_"):
                raise ValueError(f"unexpected choice {key!r}")
            category_id = int(key[3:])
            if category_id not in {c.id for c in candidates}:
                raise ValueError(f"category_id {category_id} is not in the supplied options")
        except Exception as exc:
            raise ClassificationError(
                f"Jev returned an unusable classification: {exc}", usage=usage, cost_complete=True
            ) from exc
        return category_id, usage

    def classify(self, title: str) -> ClassificationResult:
        return self._classify_recursive(title, self._choice, "Jev classification", local_validation=True)


def usd(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000 * price_per_million


def openai_cost(usage: Usage, prices: dict[str, float] | None = None) -> float:
    prices = prices or OPENAI_PRICES["openai_luna"]
    cached = usage.cached_input_tokens
    writes = usage.cache_write_tokens
    regular = max(usage.input_tokens - cached - writes, 0)
    return (
        usd(regular, prices["input"])
        + usd(cached, prices["cached_input"])
        + usd(writes, prices["cache_write"])
        + usd(usage.output_tokens, prices["output"])
    )


def deepseek_cost(usage: Usage, prices: dict[str, float]) -> float:
    return (
        usd(usage.cache_hit_tokens, prices["cache_hit_input"])
        + usd(usage.cache_miss_tokens, prices["cache_miss_input"])
        + usd(usage.output_tokens, prices["output"])
    )


def known_cost_summary(measured_total: float, complete_attempts_total: float, stats: Stats) -> dict[str, Any]:
    return {
        "known_total_usd": measured_total,
        "complete_cost_attempts_total_usd": complete_attempts_total,
        "known_cost_per_known_attempt_usd": (
            complete_attempts_total / stats.known_cost_attempts if stats.known_cost_attempts else 0.0
        ),
        "known_cost_lower_bound_per_attempt_usd": measured_total / stats.samples if stats.samples else 0.0,
        "unknown_cost_attempts": stats.unknown_cost_attempts,
    }


def stats_common(stats: Stats) -> dict[str, Any]:
    samples = stats.samples
    return {
        "samples": samples,
        "successes": stats.successes,
        "wrong_predictions": stats.wrong_predictions,
        "api_errors": stats.api_errors,
        "failures": stats.failures,
        "success_rate": stats.successes / samples if samples else 0.0,
        "timing": {
            "total_seconds": stats.total_seconds,
            "seconds_per_item": stats.total_seconds / samples if samples else 0.0,
        },
        "usage": {
            "input_tokens": stats.usage.input_tokens,
            "cached_input_tokens": stats.usage.cached_input_tokens,
            "cache_write_tokens": stats.usage.cache_write_tokens,
            "cache_hit_tokens": stats.usage.cache_hit_tokens,
            "cache_miss_tokens": stats.usage.cache_miss_tokens,
            "output_tokens": stats.usage.output_tokens,
            "api_requests_with_usage": stats.usage.requests,
            "api_requests_per_item": stats.usage.requests / samples if samples else 0.0,
        },
        "cost_accounting": {
            "known_cost_attempts": stats.known_cost_attempts,
            "unknown_cost_attempts": stats.unknown_cost_attempts,
            "known_cost_coverage": stats.known_cost_attempts / samples if samples else 0.0,
            "note": (
                "Measured usage is retained whenever available. If unknown_cost_attempts is non-zero, "
                "known_total_usd is a lower bound on true cost."
            ),
        },
        "error_examples": stats.error_examples,
    }


def strategy_output(provider: str, strategy: str, stats: Stats) -> dict[str, Any]:
    result = stats_common(stats)
    result["strategy"] = strategy
    if provider in OPENAI_MODELS:
        prices = OPENAI_PRICES[provider]
        result["model"] = OPENAI_MODELS[provider]
        result["reasoning_effort"] = OPENAI_REASONING_EFFORT
        result["cost"] = {
            **known_cost_summary(openai_cost(stats.usage, prices), openai_cost(stats.complete_cost_usage, prices), stats),
            "prices_usd_per_1m_tokens": prices,
        }
    elif provider == "deepseek_flash":
        result["model"] = DEEPSEEK_MODEL
        result["cost"] = {
            "peak": {
                **known_cost_summary(
                    deepseek_cost(stats.usage, DEEPSEEK_PEAK_PRICES),
                    deepseek_cost(stats.complete_cost_usage, DEEPSEEK_PEAK_PRICES),
                    stats,
                ),
                "prices_usd_per_1m_tokens": DEEPSEEK_PEAK_PRICES,
            },
            "off_peak": {
                **known_cost_summary(
                    deepseek_cost(stats.usage, DEEPSEEK_OFFPEAK_PRICES),
                    deepseek_cost(stats.complete_cost_usage, DEEPSEEK_OFFPEAK_PRICES),
                    stats,
                ),
                "prices_usd_per_1m_tokens": DEEPSEEK_OFFPEAK_PRICES,
            },
        }
    else:
        result["model"] = TYPESAFE_MODEL
        result["cost"] = {
            **known_cost_summary(
                usd(stats.usage.input_tokens, TYPESAFE_INPUT_PRICE),
                usd(stats.complete_cost_usage.input_tokens, TYPESAFE_INPUT_PRICE),
                stats,
            ),
            "input_price_usd_per_1m_tokens": TYPESAFE_INPUT_PRICE,
            "output_price_usd_per_1m_tokens": 0.0,
        }
    return result


def build_output(
    stats: dict[str, Stats],
    args: argparse.Namespace,
    test_count: int,
    prediction_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "number_samples": args.number_samples,
            "full_dataset_iterations": args.number_samples,
            "classifications_per_strategy": args.number_samples * test_count,
            "input_file": str(args.input_file),
            "category_file": str(args.category_file),
            "tests_in_input_file": test_count,
            "multilingual_prompt_note": MULTILINGUAL_NOTE,
            "sampling": (
                "number_samples is the number of complete dataset iterations. In every iteration, "
                "each strategy classifies every row in input_file exactly once."
            ),
        },
        "models": {
            provider: {
                strategy: strategy_output(provider, strategy, stats[f"{provider}.{strategy}"])
                for strategy in strategies if f"{provider}.{strategy}" in stats
            }
            for provider, strategies in STRATEGIES.items()
            if any(f"{provider}.{strategy}" in stats for strategy in strategies)
        },
        # Intentionally separate from aggregate statistics so consumers can drop this
        # key entirely when they only need summary metrics.
        "prediction_log": prediction_log or [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--number-samples",
        type=int,
        required=True,
        help="Number of complete passes over input.csv. Every strategy classifies every row once per pass.",
    )
    parser.add_argument("--input-file", type=Path, default=Path("input.csv"), help="Default: input.csv")
    parser.add_argument("--output-file", type=Path, default=Path("output.json"), help="Default: output.json")
    parser.add_argument(
        "--category-file", type=Path, default=Path("category_input.json"), help="Generated COICOP category tree"
    )
    parser.add_argument(
        "--classifiers", nargs="+", choices=CLASSIFIER_CHOICES, default=CLASSIFIER_CHOICES,
        metavar="MODEL.STRATEGY",
        help="Run selected pairs (default: all). Choices: " + ", ".join(CLASSIFIER_CHOICES),
    )
    return parser.parse_args()


def create_classifiers(categories: list[Category], keys: list[str] | tuple[str, ...]) -> list[Classifier]:
    """Create selected classifiers once, preserving order and removing duplicates."""
    classifiers: list[Classifier] = []
    for key in dict.fromkeys(keys):
        provider, strategy = key.split(".")
        if provider in OPENAI_MODELS:
            classifiers.append(OpenAIClassifier(categories, strategy, provider))
        elif provider == "deepseek_flash":
            classifiers.append(DeepSeekClassifier(categories, strategy))
        else:
            classifiers.append(JevClassifier(categories))
    return classifiers


def category_output(category: Category) -> dict[str, Any]:
    return {"category_id": category.id, "code": category.code, "title": category.title}


def classify_test(
    classifier: Classifier, test: TestCase, by_id: dict[int, Category], stats: Stats
) -> dict[str, Any]:
    """Measure one attempt, update its statistics, and return its prediction log."""
    expected = by_id[test.category_id]
    predicted: Category | None
    error: dict[str, str] | None = None
    requests: int | None
    started = time.perf_counter()
    try:
        result = classifier.classify(test.title)
        elapsed = time.perf_counter() - started
        predicted = by_id[result.category_id]
        stats.record_prediction(elapsed, result.usage, expected, predicted)
        correct = predicted.id == expected.id
        requests = result.usage.requests
        status = "correct" if correct else "wrong_prediction"
        console_status = "OK" if correct else f"FAIL predicted={predicted.id}"
        print(f"  {classifier.key}: {console_status} ({elapsed:.3f}s, requests={requests})")
    except Exception as exc:
        elapsed = time.perf_counter() - started
        stats.record_error(elapsed, exc)
        predicted = None
        correct = False
        status = "api_error"
        error_usage = exc.usage if isinstance(exc, ClassificationError) else None
        requests = error_usage.requests if error_usage is not None else None
        error = {"type": type(exc).__name__, "message": str(exc)}
        print(f"  {classifier.key}: ERROR {type(exc).__name__}: {exc}")

    return {
        "model": classifier.provider,
        "strategy": classifier.strategy,
        "classifier": classifier.key,
        "title": test.title,
        "expected": category_output(expected),
        "predicted": category_output(predicted) if predicted is not None else None,
        "correct": correct,
        "status": status,
        "elapsed_seconds": elapsed,
        "api_requests": requests,
        "error": error,
    }


def run_benchmark(
    classifiers: list[Classifier], tests: list[TestCase], categories: list[Category], iterations: int
) -> tuple[dict[str, Stats], list[dict[str, Any]]]:
    """Run every classifier on every row for each full dataset iteration."""
    by_id = {c.id: c for c in categories}
    stats = {classifier.key: Stats() for classifier in classifiers}
    prediction_log: list[dict[str, Any]] = []
    attempts_per_strategy = iterations * len(tests)
    for iteration in range(1, iterations + 1):
        print(
            f"Starting full dataset iteration {iteration}/{iterations} "
            f"({len(tests)} rows; {attempts_per_strategy} total classifications per strategy)."
        )
        for row_index, test in enumerate(tests, start=1):
            overall_sample = (iteration - 1) * len(tests) + row_index
            print(
                f"iteration={iteration}/{iterations} row={row_index}/{len(tests)} "
                f"sample={overall_sample}/{attempts_per_strategy} "
                f"expected_id={test.category_id} title={test.title!r}"
            )
            for classifier in classifiers:
                prediction_log.append(
                    {
                        "iteration": iteration,
                        "sample_index": row_index,
                        "overall_sample_index": overall_sample,
                        **classify_test(classifier, test, by_id, stats[classifier.key]),
                    }
                )
    return stats, prediction_log


def write_output(output: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote benchmark results to {path}")


def main() -> None:
    args = parse_args()
    if args.number_samples <= 0:
        raise ValueError("--number-samples must be greater than zero.")

    categories = load_categories(args.category_file)
    tests = load_tests(args.input_file, {c.id for c in categories})
    classifiers = create_classifiers(categories, args.classifiers)
    stats, prediction_log = run_benchmark(classifiers, tests, categories, args.number_samples)
    output = build_output(stats, args, len(tests), prediction_log)
    write_output(output, args.output_file)


if __name__ == "__main__":
    main()
