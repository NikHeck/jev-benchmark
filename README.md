This repo includes a benchmark comparing Jev, OpenAI and Deepseek models for classification of household expenses.

Copy `.env.template` to `.env`, add your API keys, then run `source .env`.

```bash
# Repeat the experiment with one pass over the dataset:
uv run python benchmark.py --number-samples 1

# Try your own input CSV:
uv run python benchmark.py --number-samples 1 --input-file path/to/your/input.csv
```
