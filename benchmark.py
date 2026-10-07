#!/usr/bin/env python3
"""Benchmark direct and recursive COICOP classification.

Compare OpenAI, DeepSeek and Jev.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

RESPONSES_REASONING_EFFORT = "none"
OPENAI_DECISIONS_INPUT_PRICE = 0.10
REQUEST_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class ProviderConfig:
    """Provider settings shared by requests, strategy selection and reports."""

    key: str
    name: str
    model: str
    api: str
    api_key_env: str
    strategies: tuple[str, ...]
    base_url: str | None = None
    reasoning_effort: str | None = None


PROVIDERS = {
    config.key: config
    for config in (
        ProviderConfig(
            key="openai_luna",
            name="OpenAI",
            model=os.getenv("OPENAI_LUNA_MODEL", "gpt-6-luna"),
            api="responses",
            api_key_env="OPENAI_API_KEY",
            strategies=("direct", "recursive"),
            reasoning_effort=RESPONSES_REASONING_EFFORT,
        ),
        ProviderConfig(
            key="openai_luna_decisions",
            name="OpenAI Decisions",
            model=os.getenv("OPENAI_LUNA_MODEL", "gpt-6-luna"),
            api="decisions",
            api_key_env="OPENAI_API_KEY",
            strategies=("recursive", "recursive_subtree"),
        ),
        ProviderConfig(
            key="openai_sol",
            name="OpenAI",
            model=os.getenv("OPENAI_SOL_MODEL", "gpt-6.1-sol"),
            api="responses",
            api_key_env="OPENAI_API_KEY",
            strategies=("direct", "recursive"),
            reasoning_effort="low",
        ),
        ProviderConfig(
            key="deepseek_flash",
            name="DeepSeek",
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            api="responses",
            api_key_env="DEEPSEEK_API_KEY",
            strategies=("direct", "recursive"),
            base_url="https://api.deepseek.com",
            reasoning_effort=RESPONSES_REASONING_EFFORT,
        ),
        ProviderConfig(
            key="typesafe_jev",
            name="Jev",
            model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
            api="jev",
            api_key_env="TYPESAFE_API_KEY",
            strategies=("recursive", "recursive_subtree"),
        ),
    )
}

# USD per 1M tokens. Verify before long benchmark runs;
# provider prices can change.
OPENAI_PRICES = {
    "openai_luna": {
        "input": 0.10,
        "cached_input": 0.01,
        "cache_write": 0.125,
        "output": 0.50,
    },
    "openai_sol": {
        "input": 2.00,
        "cached_input": 0.10,
        "cache_write": 2.50,
        "output": 10.00,
    },
}
STRATEGIES = {key: config.strategies for key, config in PROVIDERS.items()}
CLASSIFIER_CHOICES = tuple(
    f"{provider}.{strategy}"
    for provider, strategies in STRATEGIES.items()
    for strategy in strategies
)
DEEPSEEK_PEAK_PRICES = {
    "cache_hit_input": 0.006,
    "cache_miss_input": 0.30,
    "output": 1.20,
}
DEEPSEEK_OFFPEAK_PRICES = {
    "cache_hit_input": 0.003,
    "cache_miss_input": 0.15,
    "output": 0.60,
}
TYPESAFE_INPUT_PRICE = 0.042

MULTILINGUAL_NOTE = (
    "The item title may be in any language and is not necessarily English. "
    "Interpret the title in its original language before choosing "
    "the COICOP category."
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

    def to_output(self) -> dict[str, Any]:
        return {
            "category_id": self.id,
            "code": self.code,
            "title": self.title,
        }


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

    def add(self, other: Usage) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class ClassificationResult:
    category_id: int
    usage: Usage


class ClassificationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        usage: Usage | None = None,
        cost_complete: bool = False,
    ) -> None:
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
    complete_cost_usage: Usage = field(
        default_factory=lambda: Usage(requests=0)
    )
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
        *,
        correct: bool,
    ) -> None:
        if correct:
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

    def to_output(self) -> dict[str, Any]:
        samples = self.samples
        return {
            "samples": samples,
            "successes": self.successes,
            "wrong_predictions": self.wrong_predictions,
            "api_errors": self.api_errors,
            "failures": self.failures,
            "success_rate": self.successes / samples if samples else 0.0,
            "timing": {
                "total_seconds": self.total_seconds,
                "seconds_per_item": self.total_seconds / samples
                if samples
                else 0.0,
            },
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "cached_input_tokens": self.usage.cached_input_tokens,
                "cache_write_tokens": self.usage.cache_write_tokens,
                "cache_hit_tokens": self.usage.cache_hit_tokens,
                "cache_miss_tokens": self.usage.cache_miss_tokens,
                "output_tokens": self.usage.output_tokens,
                "api_requests_with_usage": self.usage.requests,
                "api_requests_per_item": self.usage.requests / samples
                if samples
                else 0.0,
            },
            "cost_accounting": {
                "known_cost_attempts": self.known_cost_attempts,
                "unknown_cost_attempts": self.unknown_cost_attempts,
                "known_cost_coverage": self.known_cost_attempts / samples
                if samples
                else 0.0,
                "descriptions": {
                    "known_cost_attempts": (
                        "Number of classification attempts whose full cost is "
                        "known, regardless of prediction correctness or "
                        "errors. One attempt classifies one item and may "
                        "make multiple API requests."
                    ),
                    "unknown_cost_attempts": (
                        "Number of classification attempts whose full cost "
                        "could not be determined. Available partial usage is "
                        "still retained."
                    ),
                    "known_cost_coverage": (
                        "known_cost_attempts / samples. Fraction of all "
                        "classification attempts with a fully known cost, "
                        "from 0 to 1. Returns 0 when there are no attempts; "
                        "coverage is unavailable in that case."
                    ),
                },
                "note": (
                    "Measured usage is retained whenever available. "
                    "If unknown_cost_attempts is non-zero, "
                    "known_total_usd is a lower bound on true cost."
                ),
            },
            "error_examples": self.error_examples,
        }


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


def build_classification_instructions(*, recursive: bool) -> str:
    """Task wording shared by JSON generation and native Choice APIs."""
    instructions = (
        "Classify the expense or product title into exactly one "
        "COICOP leaf category in the supplied taxonomy. "
        "Choose only from the supplied category options. "
        f"{MULTILINGUAL_NOTE}"
    )
    if recursive:
        instructions += (
            " In this hierarchical step, choose the available branch "
            "that best matches the item. Intermediate categories guide "
            "the traversal; continue until a category with no children "
            "is reached."
        )
    return instructions


def build_current_category_context(current: Category) -> str:
    return (
        f"CURRENT COICOP CATEGORY: {current.id} | "
        f"{current.code} | {current.title}"
    )


def build_category_prompt(categories: list[Category]) -> str:
    rows = ["AVAILABLE COICOP CATEGORIES (id | code | title):"]
    rows.extend(f"{c.id} | {c.code} | {c.title}" for c in categories)
    return "\n".join(rows)


def build_choice_instructions(
    current: Category | None,
    *,
    recursive: bool,
    subtree_context: str | None = None,
) -> str:
    """Task and tree context shared by the native Choice APIs."""
    instructions = build_classification_instructions(recursive=recursive)
    if current is not None:
        instructions += "\n" + build_current_category_context(current)
    if subtree_context is not None:
        instructions += (
            "\nUse the tree below as context to compare the available "
            "options. Choose only from the supplied choice options "
            "for this step; deeper descendants are context only.\n"
            + subtree_context
        )
    return instructions


def coicop_sort_key(code: str) -> tuple[int, ...]:
    return tuple(int(part) for part in code.split("."))


def load_categories(path: Path) -> list[Category]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("category_input.json must be a non-empty JSON list.")

    categories = [
        Category(
            id=item["id"],
            code=item["code"],
            title=item["title"],
            parent_id=item["parent_id"],
            level=item["level"],
            children_ids=tuple(item["children_ids"]),
            is_leaf=item["is_leaf"],
        )
        for item in raw
    ]
    if [c.id for c in categories] != list(range(len(categories))):
        raise ValueError(
            "Category ids must start at 0 and increase by 1 without gaps."
        )
    if len({c.code for c in categories}) != len(categories):
        raise ValueError("COICOP codes must be unique.")

    by_id = {c.id: c for c in categories}
    for c in categories:
        if c.parent_id is not None and c.parent_id not in by_id:
            raise ValueError(
                f"Category {c.id} has unknown parent_id {c.parent_id}."
            )
        for child_id in c.children_ids:
            if child_id not in by_id:
                raise ValueError(
                    f"Category {c.id} has unknown child id {child_id}."
                )
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
                "input CSV must contain a header row with columns: "
                "category_id,title."
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
                "input CSV is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        category_id_column = normalized["category_id"]
        title_column = normalized["title"]
        for row_number, row in enumerate(reader, start=2):
            raw_category_id = (row.get(category_id_column) or "").strip()
            title = (row.get(title_column) or "").strip()
            if not raw_category_id and not title:
                continue
            if not raw_category_id:
                raise ValueError(
                    f"input CSV row {row_number} has an empty category_id."
                )
            try:
                category_id = int(raw_category_id)
            except ValueError as exc:
                raise ValueError(
                    f"input CSV row {row_number} has a non-integer "
                    "category_id: "
                    f"{raw_category_id!r}."
                ) from exc
            if category_id not in valid_ids:
                raise ValueError(
                    f"input CSV row {row_number} references unknown "
                    f"category_id {category_id}."
                )
            if not title:
                raise ValueError(
                    f"input CSV row {row_number} has an empty title."
                )
            tests.append(TestCase(category_id=category_id, title=title))

    if not tests:
        raise ValueError("input CSV must contain at least one test row.")
    return tests


@dataclass
class CategoryTree:
    """Shared taxonomy indexes, traversal and subtree rendering."""

    categories: list[Category]
    by_id: dict[int, Category] = field(init=False)
    leaves: list[Category] = field(init=False)
    roots: list[Category] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {c.id: c for c in self.categories}
        self.leaves = [c for c in self.categories if not c.children_ids]
        self.roots = sorted(
            (c for c in self.categories if c.parent_id is None),
            key=lambda c: coicop_sort_key(c.code),
        )

    def children(self, current: Category) -> list[Category]:
        children = [self.by_id[cid] for cid in current.children_ids]
        children.sort(key=lambda c: coicop_sort_key(c.code))
        return children

    def subtree_context(self, current: Category | None) -> str:
        rows = [
            "COICOP SUBTREE CONTEXT (id | code | title; "
            "indentation shows parent/child relationships):"
        ]

        def append_category(category: Category, depth: int) -> None:
            rows.append(
                f"{'  ' * depth}{category.id} | "
                f"{category.code} | {category.title}"
            )
            for child in self.children(category):
                append_category(child, depth + 1)

        for root in self.roots if current is None else [current]:
            append_category(root, 0)
        return "\n".join(rows)


class ChoiceAdapter(Protocol):
    """Make one decision and report its usage, without traversing the tree."""

    config: ProviderConfig

    def choose(
        self,
        title: str,
        candidates: list[Category],
        current: Category | None,
        *,
        recursive: bool,
        subtree_context: str | None = None,
    ) -> ClassificationResult: ...


@dataclass
class TreeClassifier:
    """Run a classification strategy independently of its API format."""

    tree: CategoryTree
    adapter: ChoiceAdapter
    strategy: str

    def __post_init__(self) -> None:
        if self.strategy not in self.adapter.config.strategies:
            raise ValueError(
                f"Unsupported {self.adapter.config.name} strategy: "
                f"{self.strategy}"
            )

    @property
    def provider(self) -> str:
        return self.adapter.config.key

    @property
    def key(self) -> str:
        return f"{self.provider}.{self.strategy}"

    def _choose(
        self,
        title: str,
        candidates: list[Category],
        current: Category | None,
    ) -> ClassificationResult:
        result = self.adapter.choose(
            title,
            candidates,
            current,
            recursive=self.strategy != "direct",
            subtree_context=self.tree.subtree_context(current)
            if self.strategy == "recursive_subtree"
            else None,
        )
        valid_ids = [c.id for c in candidates]
        if (
            type(result.category_id) is not int
            or result.category_id not in valid_ids
        ):
            raise ClassificationError(
                f"{self.adapter.config.name} returned category_id "
                f"{result.category_id!r} which is not one of {valid_ids}",
                usage=result.usage,
                cost_complete=True,
            )
        return result

    def classify(self, title: str) -> ClassificationResult:
        """Walk the tree and retain billed usage if a later decision fails."""
        if self.strategy == "direct":
            return self._choose(title, self.tree.leaves, None)

        total = Usage(requests=0)
        try:
            result = self._choose(title, self.tree.roots, None)
            total.add(result.usage)
            selected = self.tree.by_id[result.category_id]
            while selected.children_ids:
                result = self._choose(
                    title, self.tree.children(selected), selected
                )
                total.add(result.usage)
                selected = self.tree.by_id[result.category_id]
            return ClassificationResult(selected.id, total)
        except ClassificationError as exc:
            if exc.usage is not None:
                total.add(exc.usage)
            raise ClassificationError(
                str(exc), usage=total, cost_complete=exc.cost_complete
            ) from exc
        except Exception as exc:
            if total.requests > 0:
                raise ClassificationError(
                    f"{self.adapter.config.name} recursive classification "
                    "failed after partial billed usage: "
                    f"{exc}",
                    usage=total,
                    cost_complete=False,
                ) from exc
            raise


def create_openai_client(config: ProviderConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=require_env(config.api_key_env),
        base_url=config.base_url,
        max_retries=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


class ResponsesAdapter:
    """Make one structured-output decision using OpenAI or DeepSeek."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.client = create_openai_client(config)

    def choose(
        self,
        title: str,
        candidates: list[Category],
        current: Category | None,
        *,
        recursive: bool,
        subtree_context: str | None = None,
    ) -> ClassificationResult:
        valid_ids = [c.id for c in candidates]
        prompt = build_category_prompt(candidates)
        if current is not None:
            prompt = build_current_category_context(current) + "\n" + prompt
        instructions = build_classification_instructions(
            recursive=recursive
        ) + (
            " Return JSON only, with exactly one integer field: "
            '{"category_id": 123}.'
        )
        response = self.client.responses.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            instructions=instructions,
            input=f"{prompt}\n\nTITLE TO CLASSIFY:\n{title}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "coicop_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "category_id": {
                                "type": "integer",
                                "enum": valid_ids,
                            }
                        },
                        "required": ["category_id"],
                        "additionalProperties": False,
                    },
                }
            },
            max_output_tokens=128,
            store=False,
        )
        usage = self._response_usage(response)
        try:
            if response.status != "completed":
                raise ValueError(f"response status is {response.status!r}")
            result = json.loads(response.output_text)
            if not isinstance(result, dict) or set(result) != {"category_id"}:
                raise ValueError("expected exactly one category_id field")
            category_id = result["category_id"]
            if type(category_id) is not int:
                raise ValueError("category_id must be an integer")
            if category_id not in valid_ids:
                raise ValueError(
                    f"category_id {category_id} is not one of {valid_ids}"
                )
        except (ValueError, TypeError) as exc:
            raise ClassificationError(
                f"{self.config.name} returned an unusable classification: "
                f"{exc}",
                usage=usage,
                cost_complete=True,
            ) from exc
        return ClassificationResult(category_id, usage)

    def _response_usage(self, response: Any) -> Usage:
        u = response.usage
        details = getattr(u, "input_tokens_details", None)
        input_tokens = int(getattr(u, "input_tokens", 0) or 0)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        if self.config.key == "deepseek_flash":
            return Usage(
                input_tokens=input_tokens,
                cache_hit_tokens=cached,
                cache_miss_tokens=input_tokens - cached,
                output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            )
        return Usage(
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            cache_write_tokens=int(
                getattr(details, "cache_write_tokens", 0) or 0
            ),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        )


class DecisionsAdapter:
    """Evaluate native choice questions through the Decisions preview API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.client = create_openai_client(config)

    def choose(
        self,
        title: str,
        candidates: list[Category],
        current: Category | None,
        *,
        recursive: bool,
        subtree_context: str | None = None,
    ) -> ClassificationResult:
        if len(candidates) > 255:
            raise ClassificationError(
                "OpenAI Decisions supports at most 255 options; "
                f"this node has {len(candidates)}.",
                usage=Usage(requests=0),
                cost_complete=True,
            )
        instructions = build_choice_instructions(
            current,
            recursive=recursive,
            subtree_context=subtree_context,
        )
        # The installed SDK predates client.decisions. Its public HTTP
        # interface keeps SDK authentication, timeouts and retry handling.
        response = self.client.post(
            "/decisions",
            cast_to=dict,
            body={
                "model": self.config.model,
                "input": title,
                "questions": [
                    {
                        "type": "choice",
                        "name": "category",
                        "instructions": instructions,
                        "choices": [
                            {
                                "value": f"id_{c.id}",
                                "description": f"COICOP {c.code}: {c.title}",
                            }
                            for c in candidates
                        ],
                    }
                ],
            },
        )
        u = response.get("usage") if isinstance(response, dict) else None
        if (
            not isinstance(u, dict)
            or type(u.get("input_tokens")) is not int
            or u["input_tokens"] < 0
        ):
            raise ClassificationError(
                "OpenAI Decisions returned missing or invalid input usage",
                usage=Usage(requests=1),
                cost_complete=False,
            )
        usage = Usage(input_tokens=u["input_tokens"])
        try:
            details = u.get("input_tokens_details") or {}
            if not isinstance(details, dict):
                raise ValueError("input_tokens_details must be an object")
            usage.cached_input_tokens = int(
                details.get("cached_tokens", 0) or 0
            )
            usage.cache_write_tokens = int(
                details.get("cache_write_tokens", 0) or 0
            )
            usage.output_tokens = int(u.get("output_tokens", 0) or 0)
            answers = response["answers"]
            if not isinstance(answers, list) or len(answers) != 1:
                raise ValueError("expected exactly one category answer")
            answer = answers[0]
            if not isinstance(answer, dict):
                raise ValueError("expected a choice answer object")
            if answer.get("type") != "choice":
                raise ValueError(
                    f"unexpected answer type {answer.get('type')!r}"
                )
            if answer.get("name") != "category":
                raise ValueError("answer name must be 'category'")
            choices = {f"id_{c.id}": c.id for c in candidates}
            key = answer.get("choice")
            if not isinstance(key, str) or key not in choices:
                raise ValueError(
                    f"choice {key!r} is not in the supplied options"
                )
            category_id = choices[key]
        except (ValueError, TypeError, KeyError) as exc:
            raise ClassificationError(
                "OpenAI Decisions returned an unusable classification: "
                f"{exc}",
                usage=usage,
                cost_complete=True,
            ) from exc
        return ClassificationResult(category_id, usage)


class JevAdapter:
    """Make one native choice decision using Jev."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        require_env(config.api_key_env)
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

        self._choice_type = Choice
        self.client = TypeSafeClient(
            retry=RetryPolicy(max_retries=0), timeout=REQUEST_TIMEOUT_SECONDS
        )

    def choose(
        self,
        title: str,
        candidates: list[Category],
        current: Category | None,
        *,
        recursive: bool,
        subtree_context: str | None = None,
    ) -> ClassificationResult:
        if len(candidates) > 255:
            raise ClassificationError(
                "Jev Choice supports at most 255 options; "
                f"this node has {len(candidates)}.",
                usage=Usage(requests=0),
                cost_complete=True,
            )
        criteria = {
            f"id_{c.id}": f"COICOP {c.code}: {c.title}" for c in candidates
        }
        instructions = build_choice_instructions(
            current,
            recursive=recursive,
            subtree_context=subtree_context,
        )
        response = self.client.system_one(
            model=self.config.model,
            state=title,
            questions={
                "category": self._choice_type(
                    instructions=instructions, criteria=criteria
                )
            },
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
                raise ValueError(
                    f"category_id {category_id} is not in the supplied options"
                )
        except Exception as exc:
            raise ClassificationError(
                f"Jev returned an unusable classification: {exc}",
                usage=usage,
                cost_complete=True,
            ) from exc
        return ClassificationResult(category_id, usage)


def usd(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000 * price_per_million


def openai_decisions_cost(usage: Usage) -> float:
    """Decisions charges only input tokens, without separate cache charges."""
    return usd(usage.input_tokens, OPENAI_DECISIONS_INPUT_PRICE)


def openai_cost(usage: Usage, prices: dict[str, float]) -> float:
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


def known_cost_summary(
    measured_total_usd: float, complete_cost_total_usd: float, stats: Stats
) -> dict[str, Any]:
    return {
        "known_total_usd": measured_total_usd,
        "complete_cost_attempts_total_usd": complete_cost_total_usd,
        "known_cost_per_known_attempt_usd": (
            complete_cost_total_usd / stats.known_cost_attempts
            if stats.known_cost_attempts
            else 0.0
        ),
        "known_cost_lower_bound_per_attempt_usd": measured_total_usd
        / stats.samples
        if stats.samples
        else 0.0,
        "unknown_cost_attempts": stats.unknown_cost_attempts,
        "descriptions": {
            "known_total_usd": (
                "Cost in USD calculated from all reported usage, including "
                "partial usage from attempts with an unknown total cost. "
                "A lower bound on total cost when unknown_cost_attempts is "
                "non-zero."
            ),
            "complete_cost_attempts_total_usd": (
                "Total cost in USD of attempts whose full cost is known, "
                "including correct predictions, wrong predictions, and "
                "errors with complete usage. Excludes all usage from "
                "attempts with an unknown total cost."
            ),
            "known_cost_per_known_attempt_usd": (
                "complete_cost_attempts_total_usd / known_cost_attempts. "
                "Average cost in USD among attempts whose full cost is "
                "known. Returns 0 when there are no such attempts; "
                "the average is unavailable in that case."
            ),
            "known_cost_lower_bound_per_attempt_usd": (
                "known_total_usd / samples. Lower bound on the average "
                "cost in USD across all classification attempts, including "
                "those with missing usage. Can be lower or higher than "
                "known_cost_per_known_attempt_usd because the denominators "
                "differ. Returns 0 when there are no attempts; the average "
                "is unavailable in that case."
            ),
            "unknown_cost_attempts": (
                "Number of classification attempts whose full cost could "
                "not be determined. Their reported partial usage still "
                "contributes to known_total_usd. One attempt classifies "
                "one item and may make multiple API requests."
            ),
        },
    }


def strategy_output(
    provider: str, strategy: str, stats: Stats
) -> dict[str, Any]:
    result = stats.to_output()
    result["strategy"] = strategy
    config = PROVIDERS[provider]
    result["model"] = config.model
    if config.reasoning_effort is not None:
        result["reasoning_effort"] = config.reasoning_effort
    if provider in OPENAI_PRICES:
        prices = OPENAI_PRICES[provider]
        result["cost"] = {
            **known_cost_summary(
                openai_cost(stats.usage, prices),
                openai_cost(stats.complete_cost_usage, prices),
                stats,
            ),
            "prices_usd_per_1m_tokens": prices,
        }
    elif provider == "openai_luna_decisions":
        result["api"] = "decisions"
        result["cost"] = {
            **known_cost_summary(
                openai_decisions_cost(stats.usage),
                openai_decisions_cost(stats.complete_cost_usage),
                stats,
            ),
            "input_price_usd_per_1m_tokens": OPENAI_DECISIONS_INPUT_PRICE,
            "output_price_usd_per_1m_tokens": 0.0,
        }
    elif provider == "deepseek_flash":
        result["cost"] = {
            "peak": {
                **known_cost_summary(
                    deepseek_cost(stats.usage, DEEPSEEK_PEAK_PRICES),
                    deepseek_cost(
                        stats.complete_cost_usage, DEEPSEEK_PEAK_PRICES
                    ),
                    stats,
                ),
                "prices_usd_per_1m_tokens": DEEPSEEK_PEAK_PRICES,
            },
            "off_peak": {
                **known_cost_summary(
                    deepseek_cost(stats.usage, DEEPSEEK_OFFPEAK_PRICES),
                    deepseek_cost(
                        stats.complete_cost_usage, DEEPSEEK_OFFPEAK_PRICES
                    ),
                    stats,
                ),
                "prices_usd_per_1m_tokens": DEEPSEEK_OFFPEAK_PRICES,
            },
        }
    else:
        result["cost"] = {
            **known_cost_summary(
                usd(stats.usage.input_tokens, TYPESAFE_INPUT_PRICE),
                usd(
                    stats.complete_cost_usage.input_tokens,
                    TYPESAFE_INPUT_PRICE,
                ),
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
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "configuration": {
            "number_samples": args.number_samples,
            "full_dataset_iterations": args.number_samples,
            "classifications_per_strategy": args.number_samples * test_count,
            "input_file": str(args.input_file),
            "category_file": str(args.category_file),
            "tests_in_input_file": test_count,
            "multilingual_prompt_note": MULTILINGUAL_NOTE,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "classification_target": "leaf_category",
            "sampling": (
                "number_samples is the number of complete dataset "
                "iterations. In every iteration, each strategy classifies "
                "every row in input_file exactly once."
            ),
        },
        "models": {
            provider: {
                strategy: strategy_output(
                    provider, strategy, stats[f"{provider}.{strategy}"]
                )
                for strategy in strategies
                if f"{provider}.{strategy}" in stats
            }
            for provider, strategies in STRATEGIES.items()
            if any(
                f"{provider}.{strategy}" in stats for strategy in strategies
            )
        },
        # Separate from aggregate statistics so consumers can drop this
        # key entirely when they only need summary metrics.
        "prediction_log": prediction_log or [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--number-samples",
        type=int,
        required=True,
        help=(
            "Number of complete passes over input.csv. "
            "Every strategy classifies every row once per pass."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("input.csv"),
        help="Default: input.csv",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("output.json"),
        help="Default: output.json",
    )
    parser.add_argument(
        "--category-file",
        type=Path,
        default=Path("category_input.json"),
        help="Generated COICOP category tree",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=CLASSIFIER_CHOICES,
        default=CLASSIFIER_CHOICES,
        metavar="MODEL.STRATEGY",
        help="Run selected pairs (default: all). Choices: "
        + ", ".join(CLASSIFIER_CHOICES),
    )
    return parser.parse_args()


def create_classifiers(
    categories: list[Category], keys: list[str] | tuple[str, ...]
) -> list[Classifier]:
    """Create classifiers once, preserving order and removing duplicates."""
    tree = CategoryTree(categories)
    adapter_types = {
        "responses": ResponsesAdapter,
        "decisions": DecisionsAdapter,
        "jev": JevAdapter,
    }
    classifiers: list[Classifier] = []
    for key in dict.fromkeys(keys):
        provider, strategy = key.split(".")
        config = PROVIDERS.get(provider)
        if config is None:
            raise ValueError(f"Unknown provider: {provider}")
        if strategy not in config.strategies:
            raise ValueError(
                f"Unsupported {config.name} strategy: {strategy}"
            )
        adapter = adapter_types[config.api](config)
        classifiers.append(TreeClassifier(tree, adapter, strategy))
    return classifiers


def classify_test(
    classifier: Classifier,
    test: TestCase,
    by_id: dict[int, Category],
    stats: Stats,
) -> dict[str, Any]:
    """Measure an attempt, update statistics, and return its prediction log."""
    expected = by_id[test.category_id]
    predicted: Category | None
    error: dict[str, str] | None = None
    requests: int | None
    started = time.perf_counter()
    try:
        result = classifier.classify(test.title)
        elapsed = time.perf_counter() - started
        predicted = by_id[result.category_id]
        correct = predicted.id == expected.id
        stats.record_prediction(elapsed, result.usage, correct=correct)
        requests = result.usage.requests
        status = "correct" if correct else "wrong_prediction"
        console_status = "OK" if correct else f"FAIL predicted={predicted.id}"
        print(
            f"  {classifier.key}: {console_status} "
            f"({elapsed:.3f}s, requests={requests})"
        )
    except Exception as exc:  # noqa: BLE001
        # Record provider failures so remaining benchmark attempts still run.
        elapsed = time.perf_counter() - started
        stats.record_error(elapsed, exc)
        predicted = None
        correct = False
        status = "api_error"
        error_usage = (
            exc.usage if isinstance(exc, ClassificationError) else None
        )
        requests = error_usage.requests if error_usage is not None else None
        error = {"type": type(exc).__name__, "message": str(exc)}
        print(f"  {classifier.key}: ERROR {type(exc).__name__}: {exc}")

    return {
        "model": classifier.provider,
        "strategy": classifier.strategy,
        "classifier": classifier.key,
        "title": test.title,
        "expected": expected.to_output(),
        "predicted": predicted.to_output() if predicted is not None else None,
        "correct": correct,
        "status": status,
        "elapsed_seconds": elapsed,
        "api_requests": requests,
        "error": error,
    }


def run_benchmark(
    classifiers: list[Classifier],
    tests: list[TestCase],
    categories: list[Category],
    iterations: int,
) -> tuple[dict[str, Stats], list[dict[str, Any]]]:
    """Run every classifier on every row for each full dataset iteration."""
    by_id = {c.id: c for c in categories}
    stats = {classifier.key: Stats() for classifier in classifiers}
    prediction_log: list[dict[str, Any]] = []
    attempts_per_strategy = iterations * len(tests)
    for iteration in range(1, iterations + 1):
        print(
            f"Starting full dataset iteration {iteration}/{iterations} "
            f"({len(tests)} rows; {attempts_per_strategy} "
            "total classifications per strategy)."
        )
        for row_index, test in enumerate(tests, start=1):
            overall_sample = (iteration - 1) * len(tests) + row_index
            print(
                f"iteration={iteration}/{iterations} "
                f"row={row_index}/{len(tests)} "
                f"sample={overall_sample}/{attempts_per_strategy} "
                f"expected_id={test.category_id} title={test.title!r}"
            )
            for classifier in classifiers:
                prediction_log.append(
                    {
                        "iteration": iteration,
                        "sample_index": row_index,
                        "overall_sample_index": overall_sample,
                        **classify_test(
                            classifier, test, by_id, stats[classifier.key]
                        ),
                    }
                )
    return stats, prediction_log


def write_output(output: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote benchmark results to {path}")


def main() -> None:
    args = parse_args()
    if args.number_samples <= 0:
        raise ValueError("--number-samples must be greater than zero.")

    categories = load_categories(args.category_file)
    tests = load_tests(args.input_file, {c.id for c in categories})
    leaf_ids = {c.id for c in categories if not c.children_ids}
    for test in tests:
        if test.category_id not in leaf_ids:
            raise ValueError(
                f"input CSV references non-leaf category_id "
                f"{test.category_id}; expected categories must be leaves."
            )
    classifiers = create_classifiers(categories, args.classifiers)
    stats, prediction_log = run_benchmark(
        classifiers, tests, categories, args.number_samples
    )
    output = build_output(stats, args, len(tests), prediction_log)
    write_output(output, args.output_file)


if __name__ == "__main__":
    main()
