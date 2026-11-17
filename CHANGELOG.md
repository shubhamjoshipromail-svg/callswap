# Changelog

## 0.1.0 (unreleased)

First public release.

- `callswap run`: ingest, leakage-safe split, model selection, temperature calibration, abstain threshold, one sealed evaluation
  with a 95% Wilson lower bound, HTML and JSON report with per-label results.
- `callswap demo`: the whole pipeline on a bundled copy of Banking77, with no API key and no network.
- `callswap serve`: OpenAI-compatible endpoint that answers locally when confident and forwards to the original LLM otherwise;
  shadow mode; `/classify`, `/stats`, `/healthz`; configuration through environment variables.
- `callswap label`: produce teacher labels from any OpenAI-compatible endpoint with your own key.
- Three candidate models: TF-IDF + logistic regression (base install), sentence embeddings (`[embeddings]`), fine-tuned
  transformer (`[finetune]`).
- Dockerfile and compose file; CI on Python 3.11 to 3.13; image publishing workflow for tags.
- Benchmarks on Banking77, CLINC150, MASSIVE (en-US) and SMS Spam with all three models (3 of 12 pass the default bar), with gold labels standing in for the teacher.

Known gaps: no drift detection or automatic retraining, single-label classification only, no streaming responses, no benchmark
against a real LLM teacher yet.
