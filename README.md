# COICOP model benchmark

Benchmark category classification against the official **COICOP 2018** taxonomy using:

- OpenAI **GPT-6 Luna** (`gpt-6-luna`) — direct + recursive
- OpenAI **GPT-6 Sol** (`gpt-6-sol`) — direct + recursive
- DeepSeek **V4.1 Flash** (`deepseek-flash`) — direct + recursive
- TypeSafe AI **Jev** (`jev-latest`) — recursive

The project uses [`uv`](https://docs.astral.sh/uv/) for Python/dependency management.

## Experiment matrix

| Backend | Direct, all categories | Recursive tree walk |
|---|---:|---:|
| OpenAI Luna | yes | yes |
| OpenAI Sol | yes | yes |
| DeepSeek Flash | yes | yes |
| TypeSafe Jev | no | yes |

Every classification prompt explicitly states that **the item title may be in any language and is not necessarily English**.

## Setup

```bash
uv sync --dev
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export TYPESAFE_API_KEY="..."
```

Model IDs can be overridden with `OPENAI_LUNA_MODEL`, `OPENAI_SOL_MODEL`, `DEEPSEEK_MODEL`, and `TYPESAFE_MODEL`. OpenAI Luna, OpenAI Sol, and DeepSeek Flash all use `none` reasoning effort for each strategy.

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
```

Defaults:

- `--input-file input.csv`
- `--output-file output.json`
- `--category-file category_input.json`
- `--classifiers MODEL.STRATEGY ...` selects model/strategy pairs; all seven run by default. Use `--help` to see valid pairs.

`--number-samples N` means **N complete passes over the entire input CSV**. Every strategy classifies every row once per pass. For example, with 30 input rows and `--number-samples 5`, each strategy performs 150 classifications.

With `R` rows in `input.csv`, every selected strategy performs `N × R` classification attempts. Across the seven default strategies, the run performs `7 × N × R` classification attempts in total. For example, with 30 rows and `N=5`:

- OpenAI direct: 150
- OpenAI recursive: 150
- OpenAI Sol direct: 150
- OpenAI Sol recursive: 150
- DeepSeek direct: 150
- DeepSeek recursive: 150
- Jev recursive: 150
- Total across strategies: 1050

A recursive classification attempt can contain several API requests; request counts and per-item request averages are reported separately.

## Direct vs recursive classification

OpenAI and DeepSeek use the Responses API with identical classification instructions,
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

OpenAI/DeepSeek receive the entire COICOP category list and choose one category in a single API request.

### Recursive

All recursive implementations use the same tree semantics:

1. choose one root category;
2. at the selected category, choose either the **current category itself** (stop) or one of its direct children;
3. continue until the model stops or reaches a leaf.

This makes OpenAI recursive, DeepSeek recursive, and Jev recursive directly comparable at the decision-strategy level.

## Output structure

Each strategy gets its own independent statistics object:

```json
{
  "models": {
    "openai_luna": {
      "direct": { "...": "..." },
      "recursive": { "...": "..." }
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
      "recursive": { "...": "..." }
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
- DeepSeek pricing: https://api-docs.deepseek.com/quick_start/pricing/
- TypeSafe Choice: https://docs.typesafe.ai/primitives/choice
