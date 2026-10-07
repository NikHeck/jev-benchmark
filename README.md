# COICOP model benchmark

Benchmark category classification against the official **COICOP 2018** taxonomy using:

- OpenAI **GPT-6 Luna** (`gpt-6-luna`) — direct + recursive
- OpenAI **GPT-6 Luna Decisions API** (beta) — direct + recursive + recursive with subtree context
- OpenAI **GPT-6 Sol** (`gpt-6-sol`) — direct + recursive
- DeepSeek **V4.1 Flash** (`deepseek-flash`) — direct + recursive
- TypeSafe AI **Jev** (`jev-latest`) — recursive + recursive with subtree context

The project uses [`uv`](https://docs.astral.sh/uv/) for Python/dependency management.

## Experiment matrix

| Backend | Direct, all leaves | Recursive, current options | Recursive, subtree context |
|---|---:|---:|---:|
| OpenAI Luna Responses | yes | yes | no |
| OpenAI Luna Decisions (beta) | yes | yes | yes |
| OpenAI Sol | yes | yes | no |
| DeepSeek Flash | yes | yes | no |
| TypeSafe Jev | no | yes | yes |

Every classification prompt explicitly states that **the item title may be in any language and is not necessarily English**.

## Setup

```bash
uv sync --dev
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export TYPESAFE_API_KEY="..."
```

Model IDs can be overridden with `OPENAI_LUNA_MODEL`, `OPENAI_SOL_MODEL`, `DEEPSEEK_MODEL`, and `TYPESAFE_MODEL`. OpenAI Luna Responses, OpenAI Sol, and DeepSeek Flash all use `none` reasoning effort for each strategy. Decisions shares `OPENAI_LUNA_MODEL`; the beta currently supports only `gpt-6-luna` and has no reasoning-effort parameter.

`RESPONSES_REASONING_EFFORT` is the shared setting for OpenAI and DeepSeek
Responses requests and is reported for both providers. DeepSeek's
[thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/)
confirms that `reasoning.effort: "none"` disables thinking mode. This provides
a common baseline for comparing classification latency, accuracy, and cost.

All provider clients use an explicit **600-second HTTP timeout**. For recursive
classification, this applies to each API request. The timeout is recorded as
`configuration.request_timeout_seconds` in the results.

## 1. Build the COICOP category tree

```bash
uv run python build_category_input.py
```

The script downloads the official UN COICOP 2018 Excel structure and creates `category_input.json`. IDs start at 0 and increase by 1 in source order after redundant pass-through categories are collapsed. By default, only household-expenditure divisions **01–13** are included; divisions 14–15 describe NPISH/government expenditure and are not candidates for ordinary consumer expenses.

Each category contains the original requested fields plus explicit tree metadata:

```json
{
  "id": 2,
  "code": "01.1.1",
  "title": "Cereals and cereal products",
  "parent_id": 1,
  "level": 3,
  "is_optional_detail": false,
  "children_ids": [3, 4],
  "is_leaf": false
}
```

### Collapsing redundant pass-through categories

COICOP can repeat the same semantic category across adjacent levels. For example, a parent may be named `Breakfast cereals (ND)` while its only child is `Breakfast cereals`, or a standard class can be repeated as a trailing-zero subclass such as `01.2.3` -> `01.2.3.0`. Presenting both nodes would give a classifier duplicate valid answers.

The build script therefore collapses a node when **both** conditions hold:

1. it has exactly one direct child; and
2. the parent/child titles are equivalent after normalizing punctuation/whitespace and stripping a trailing COICOP durability marker (`ND`, `SD`, `D`, `S`).

Collapse is transitive. In an equivalent chain `A -> B -> C`, only the deepest category remains and the removed codes are retained as metadata:

```json
{
  "id": 42,
  "code": "01.2.3.0",
  "title": "Tea, maté and other plant-derived products for infusion",
  "collapsed_codes": ["01.2.3"],
  "parent_id": 39,
  "level": 4,
  "is_optional_detail": false,
  "children_ids": [],
  "is_leaf": true
}
```

The retained `level` is the original code depth, so the collapsed tree can intentionally skip redundant levels. The UN workbook also contains optional high-detail categories below the standard four COICOP levels; these are retained for receipt-line classification and marked with `is_optional_detail: true`. Because collapsing/filtering reassigns dense numeric IDs, regenerate or update `input.csv` after rebuilding `category_input.json`.

The canonical tree is produced once by the preparation script. `benchmark.py` requires the generated tree metadata and reads it directly.

Useful options:

The XLSX must have a header in row 1 and category rows starting at row 2. Sheet and column indexes are **1-based**. Defaults match the local workbook: sheet 1, codes in column 1 (A), titles in column 2 (B). Other columns are ignored.

```bash
uv run python build_category_input.py --help
uv run python build_category_input.py --reuse-excel
uv run python build_category_input.py --reuse-excel --sheet-number 1 --code-column 1 --title-column 2
# Include the complete institutional COICOP scope, including divisions 14-15:
uv run python build_category_input.py --include-all-divisions
```

## 2. Create `input.csv`

The benchmark input requires exactly two semantic columns: `category_id` and `title`. `category_id` is the correct numeric category ID from the freshly generated `category_input.json`. Any other CSV columns are ignored, so you can keep your own `sample_id`, notes, merchant metadata, etc. in the file. Standard CSV quoting is supported, so titles may contain commas.

```csv
category_id,title
42,Persil Color Waschmittel 4in1 Discs
17,"Bio Vollmilch 3,8% 1L"
```

This is also valid; the extra columns do not affect the benchmark:

```csv
sample_id,category_id,title,notes
1,42,Persil Color Waschmittel 4in1 Discs,difficult German receipt abbreviation
2,17,"Bio Vollmilch 3,8% 1L",control
```

Titles may be German, English, or any other language. Rows with a missing/invalid `category_id` or an empty `title` are rejected before any paid API calls are made.

## 3. Run

```bash
uv run python benchmark.py --number-samples 5
uv run python benchmark.py --number-samples 5 --classifiers openai_luna.direct openai_sol.recursive
# Compare the two Jev context variants:
uv run python benchmark.py --number-samples 5 --classifiers typesafe_jev.recursive typesafe_jev.recursive_subtree --output-file output_jev_context.json
# Run all three Luna Decisions variants:
uv run python benchmark.py --number-samples 5 --classifiers openai_luna_decisions.direct openai_luna_decisions.recursive openai_luna_decisions.recursive_subtree --output-file output_luna_decisions.json
```

Defaults:

- `--input-file input.csv`
- `--output-file output.json`
- `--category-file category_input.json`
- `--classifiers MODEL.STRATEGY ...` selects model/strategy pairs; all eleven run by default. Use `--help` to see valid pairs.

`--number-samples N` means **N complete passes over the entire input CSV**. Every strategy classifies every row once per pass. For example, with 30 input rows and `--number-samples 5`, each strategy performs 150 classifications.

With `R` rows in `input.csv`, every selected strategy performs `N × R` classification attempts. Across the eleven default strategies, the run performs `11 × N × R` classification attempts in total. For example, with 30 rows and `N=5`:

- OpenAI direct: 150
- OpenAI recursive: 150
- OpenAI Luna Decisions direct: 150
- OpenAI Luna Decisions recursive: 150
- OpenAI Luna Decisions recursive with subtree context: 150
- OpenAI Sol direct: 150
- OpenAI Sol recursive: 150
- DeepSeek direct: 150
- DeepSeek recursive: 150
- Jev recursive: 150
- Jev recursive with subtree context: 150
- Total across strategies: 1650

A recursive classification attempt can contain several API requests; request counts and per-item request averages are reported separately.

## Direct vs recursive classification

All strategies return only leaf categories in the supplied taxonomy. Leaves
are categories with no children, including the optional food detail retained
in our tree; they need not occur at the same depth. The CLI rejects input rows
whose expected category is an intermediate node before creating provider
clients. Output configuration records `classification_target: "leaf_category"`.

The standard OpenAI and DeepSeek strategies use the Responses API with identical
classification instructions,
category/title input, JSON schema (including the allowed category IDs), `none`
reasoning effort, and a 128-token output limit. Both responses are validated locally
against the same rules. DeepSeek cache-hit and cache-miss costs are derived from
`input_tokens_details.cached_tokens` and total input tokens.

DeepSeek's Responses API and JSON-schema format support are documented in its
[API reference](https://api-docs.deepseek.com/api/create-response/) and
[compatibility guide](https://api-docs.deepseek.com/guides/responses_api/).
Existing benchmark result files predate this alignment; rerun the benchmark for
results using the shared prompt and output constraints.

### Direct

OpenAI/DeepSeek receive all leaf categories and choose one in a single API
request. The JSON schema and local validation allow only leaf IDs.

### Recursive

All recursive implementations share the classification wording, multilingual
guidance, current-category context, and leaf-only target. Jev receives its
answer options through native `Choice.criteria`; Luna Decisions receives native
`choice` options; the Responses strategies receive the category list and JSON
output instructions.

All recursive implementations use the same tree semantics:

1. choose one root category;
2. at each intermediate category, choose one of its direct children;
3. finish automatically when the selected category has no children.

The standard `recursive` strategy supplies only the options for the current
decision and, after the root decision, the current category. This makes OpenAI
recursive, DeepSeek recursive, and Jev recursive directly comparable at the
decision-strategy level.

### Recursive with subtree context

`typesafe_jev.recursive_subtree` and `openai_luna_decisions.recursive_subtree`
use the same recursive walk to a leaf, but add an indented category tree
to the choice instructions:

- Before choosing a root, the context contains the entire tree, including all roots and descendants.
- At each later decision, the context contains the current category and all its descendants. Other branches and ancestors are omitted.
- The selectable options remain the roots initially, then the direct children of the current category. The current category and deeper descendants provide context and cannot be selected at that step.

Indentation follows the actual parent/child links, including when collapsed
categories skip COICOP code levels. The extra context lets the model compare
what each branch contains before descending. It uses more input tokens per request; the
benchmark reports accuracy, latency, token usage and cost independently for each
variant. Jev's 255-option Choice limit applies to selectable options, not the
number of categories described in the context.

### Luna Decisions API (beta)

The [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
evaluates typed questions at `/v1/decisions`. The benchmark asks one native
`choice` question named `category`, with the same COICOP descriptions as Jev,
and validates the returned choice locally. It adds three independent strategies:

- `openai_luna_decisions.recursive`: current-step options, matching Jev recursive.
- `openai_luna_decisions.recursive_subtree`: the same options plus the current subtree, matching Jev subtree context.
- `openai_luna_decisions.direct`: all leaf categories are supplied as choices in one request.

The [API reference](https://developers.openai.com/api/reference/resources/decisions/methods/create)
defines a list of choice options without a published category-count limit. The
direct variant sends all leaves without splitting them into batches;
preview access and acceptance of that list still depend on the API. A refusal or
an invalid answer counts as an API error, retaining any measured usage. Missing
input usage marks the cost as incomplete.

The adapter uses the OpenAI SDK's public `post` interface because the locked SDK
version predates `client.decisions`. It uses the existing `OPENAI_API_KEY`,
600-second timeout, and disabled automatic retries. Decisions does not take the
Responses API's JSON-schema, output-token-limit, or reasoning parameters.

## Output structure

Each strategy gets its own independent statistics object:

```json
{
  "models": {
    "openai_luna": {
      "direct": { "...": "..." },
      "recursive": { "...": "..." }
    },
    "openai_luna_decisions": {
      "direct": { "...": "..." },
      "recursive": { "...": "..." },
      "recursive_subtree": { "...": "..." }
    },
    "openai_sol": {
      "direct": { "...": "..." },
      "recursive": { "...": "..." }
    },
    "deepseek_flash": {
      "direct": { "...": "..." },
      "recursive": { "...": "..." }
    },
    "typesafe_jev": {
      "recursive": { "...": "..." },
      "recursive_subtree": { "...": "..." }
    }
  }
}
```

Each strategy reports:

- exact successes, wrong predictions, API errors and success rate;
- total and per-item latency;
- token usage;
- API requests and API requests per item;
- measured cost and cost coverage;
- up to five API/error examples.

### Exact category accuracy

`success_rate` is the number of exact category ID matches divided by all classification attempts. A different category, including a parent or child of the expected category, counts as a wrong prediction. API errors count as unsuccessful attempts. No partial credit is awarded for matching an ancestor or branch.

### Prediction log

`output.json` also contains a top-level `prediction_log` array. It is intentionally separate from `models`, so it can be removed or ignored without affecting aggregate benchmark statistics. There is one entry for every dataset iteration × input row × backend strategy. Each entry records the iteration and sample, item title, backend/strategy, expected category, final predicted category, correctness, elapsed time, request count, and any API error. Recursive strategies log their final category prediction; their internal hierarchy choices are not expanded into separate log rows.

Example:

```json
{
  "prediction_log": [
    {
      "iteration": 1,
      "sample_index": 1,
      "model": "openai_luna",
      "strategy": "direct",
      "title": "Espresso Pulver",
      "expected": {"category_id": 650, "code": "01.2.2.0.1", "title": "Coffee"},
      "predicted": {"category_id": 650, "code": "01.2.2.0.1", "title": "Coffee"},
      "correct": true,
      "status": "correct",
      "error": null
    }
  ]
}
```

## Cost handling

Correct and wrong model responses both count their full measured cost. If an API/parse failure still exposes usage, that usage is retained. If complete usage cannot be recovered, the attempt is marked `unknown_cost_attempts`, and `known_total_usd` is explicitly only a lower bound.

For recursive classification, successful earlier subcalls remain counted even if a later subcall fails.

Each cost block and `cost_accounting` block includes a `descriptions` map
explaining its metrics. The cost descriptions include the formulas and
denominators for both averages: the lower bound uses all classification
attempts, while the average over known costs uses only attempts whose full
cost is known. Either average can be higher. An attempt classifies one item
and may make multiple API requests.

Luna Decisions uses its own [input-only pricing](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability):
**$0.10 per 1M input tokens**, without separate cache-read, cache-write or output
charges. Its cost is reported separately from Luna Responses. Estimates use
standard, short-context pricing; regional and long-context multipliers are not
included.

DeepSeek reports **both peak and off-peak counterfactual costs** from the exact same measured token usage:

| Token type | Peak | Off-peak |
|---|---:|---:|
| Cache-hit input / 1M | $0.006 | $0.003 |
| Cache-miss input / 1M | $0.30 | $0.15 |
| Output / 1M | $1.20 | $0.60 |

Pricing constants are near the top of `benchmark.py`; verify them before important/long runs because provider pricing can change.

## Tests

```bash
uv run pytest
```

The tests are local and do not call any paid model API.

## Official references used when creating the repo

- UN COICOP 2018: https://unstats.un.org/unsd/classifications/coicop
- OpenAI models: https://developers.openai.com/api/docs/models
- OpenAI Decisions: https://developers.openai.com/api/docs/guides/decisions
- DeepSeek pricing: https://api-docs.deepseek.com/quick_start/pricing/
- TypeSafe Choice: https://docs.typesafe.ai/primitives/choice
