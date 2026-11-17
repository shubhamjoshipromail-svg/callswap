# callswap: build spec (v0, single-label routing only)

**Product sentence:** replace one expensive LLM routing/classification call with a small model that matches it on a sealed
test, falling back to the original LLM when unsure. The customer keeps their existing labels.

This spec is the contract for the builder. Build exactly this. No extra features, no UI beyond the HTML report.

## Non-negotiable rules
1. **Labels are preserved.** Never merge, rename or drop a customer's labels. Rare or confusable labels are *reported*, not changed.
2. **The sealed test is touched once.** Split first, write the manifest with hashes, and only then train. Nothing in training,
   model selection, calibration or threshold choice may read the sealed split. Code must enforce this (the sealed file is
   opened only by `evaluate`), and a test must prove it.
3. **No leakage.** Deduplicate before splitting; exact and normalised-text duplicates must land in the same split.
   If a `group` or `timestamp` column exists, split by group / by time (latest = sealed).
4. **Agreement is not correctness.** Every number is "agreement with the teacher labels". The report says so in words.
5. **Honest statistics.** Report agreement on accepted calls with a 95% Wilson lower bound, and coverage. Verdict uses the lower bound.
6. **Deterministic.** Fixed seeds; the same input gives the same run. No LLM calls anywhere in v0 except the optional teacher fallback in `serve`.
7. No network at train time except an optional one-time embedding model download.

## Layout
```
callswap/
  __init__.py
  cli.py          # `python -m callswap <cmd>`
  ingest.py       # read CSV/JSONL -> normalised records, dedupe, label report
  split.py        # train / calibration / sealed, manifest + sha256
  train.py        # candidates, model selection by cross-validation on train only
  calibrate.py    # probability calibration + abstain threshold on calibration split only
  evaluate.py     # the one sealed evaluation; acceptance verdict
  report.py       # report.html + report.json
  serve.py        # FastAPI: OpenAI-compatible endpoint with teacher fallback, shadow mode, metering
  economics.py    # savings maths
tests/
scripts/demo_banking77.py
```
Python 3.11+, a local `.venv`. Dependencies: numpy, scikit-learn, joblib, fastapi, uvicorn, httpx, pytest;
`sentence-transformers` is an optional extra (`--embeddings`), never required for tests.

## Input
CSV or JSONL with columns `text` and `label` (names overridable with `--text-col`, `--label-col`); optional `id`, `group`, `timestamp`.
`label` is what the customer's current LLM answered for that input.

## Commands
`python -m callswap run DATA --out runs/NAME [--target-agreement 0.99] [--min-coverage 0.70] [--price-per-1k-calls 0.50] [--monthly-calls 3000000] [--embeddings] [--seed 7]`
runs ingest -> split -> train -> calibrate -> evaluate -> report and exits 0 whether the verdict is PASS or FAIL (non-zero only on errors).

`python -m callswap serve runs/NAME [--teacher-url URL --teacher-model M] [--shadow]` (teacher key from env `TEACHER_API_KEY`).

## Stages
**ingest**: strip, drop empty text, normalise for duplicate detection (lowercase, collapse whitespace). Conflicting duplicates
(same normalised text, different labels) are counted and reported as "teacher inconsistency" and all copies are kept together
in one split with the majority label used for training. Refuse (clear error) if fewer than 200 rows or fewer than 2 labels.
Label report: count per label; labels with fewer than 20 rows are flagged "too rare to prove".

**split**: 60% train / 20% calibration / 20% sealed, stratified by label where possible, respecting duplicates/groups/time as above.
Write `manifest.json`: counts, seed, and sha256 of the sorted sealed ids+labels and of each split file. `evaluate` re-checks the hash and refuses to run if it differs or if `sealed_result.json` already exists.

**train**: candidates (a) TF-IDF word 1-2grams + char 3-5grams -> LogisticRegression; (b) with `--embeddings`, frozen sentence embeddings -> LogisticRegression.
Select by 5-fold stratified CV macro-F1 on **train only**. Refit the winner on train. Save `model.joblib` and `train_result.json`.

**calibrate**: fit temperature scaling (single scalar on logits, minimise NLL) on the calibration split. Then choose the lowest
confidence threshold `t` such that, on the calibration split, the Wilson 95% lower bound of agreement among examples with
max-probability >= t is >= `--target-agreement`. If none exists, pick the threshold with the best lower bound and record `target_met_on_calibration: false`.
Save the coverage/agreement curve (at least 50 thresholds).

**evaluate** (sealed, once): with the locked model, temperature and threshold: coverage, agreement on accepted, Wilson lower bound,
overall accuracy without abstaining, per-label coverage and agreement, the 20 lowest-confidence accepted disagreements (text truncated to 200 chars).
Verdict PASS iff coverage >= min-coverage AND lower bound >= target-agreement. Also emit `labels_unproven` (labels with < 20 sealed examples).

**economics**: baseline = monthly_calls * price; after = (1 - coverage) * baseline + fixed serving cost (default $20/month, flag `--serving-cost`).
Report monthly saving, percent, and the break-even monthly call volume. If saving <= 0 say so plainly.

**report**: one self-contained `report.html` (no external assets, readable on a phone, light and dark) and `report.json`.
Order: verdict line; the three headline numbers (coverage, agreement lower bound, monthly saving); what was tested and how
(split sizes, sealed hash, "sealed set used once"); coverage/agreement curve as inline SVG; per-label table; teacher
inconsistencies and rare labels; lowest-confidence disagreements; plain-language limits (agreement is not correctness; rare labels unproven; data drift not measured).

**serve**: `POST /v1/chat/completions` (OpenAI shape, non-streaming). Classify the last user message. If confidence >= threshold:
return the label as the assistant message with header `x-callswap-route: local`. Otherwise forward the original request body to the teacher
(`route: teacher`); if no teacher is configured return the local label with `x-callswap-route: local-unsure`.
`--shadow`: always forward to the teacher and return its answer, but log the local prediction and whether it agreed.
Append one JSON line per request to `runs/NAME/requests.log` (timestamp, route, confidence, latency ms, agreed if known; **never the text**).
`GET /healthz`, `GET /stats` (counts by route, shadow agreement rate).

## Tests (pytest, no network, fast)
Use a small synthetic dataset generated in the test (separable classes plus injected duplicates and noise).
- duplicates never straddle splits; time split puts the latest rows in sealed
- the sealed file is not opened before `evaluate` (monkeypatch/open-spy), and `evaluate` refuses a second run and a tampered hash
- threshold selection uses calibration only; Wilson bound matches a hand-computed value
- label set after training equals the input label set
- economics: a hand-computed case, and the non-positive-saving case
- serve: local route, teacher fallback (mock transport), shadow mode logging, log contains no text
- `run` end to end on synthetic data produces report.html and report.json with a verdict

## Demo on real public data
`scripts/demo_banking77.py`: download the Banking77 intent dataset (Hugging Face `PolyAI/banking77` raw CSVs over HTTPS; if that
fails, print the error and exit 1, do not substitute data), write `data/banking77.csv`, and run the pipeline. Gold labels stand in for the teacher; the script and README must say this is a stand-in.

## Deliverables
Working code, passing tests, `README.md` (what it is, quick start, what is and is not proven), and `BUILD_REPORT.md`:
what was built, test output, the Banking77 numbers exactly as produced, anything in this spec you could not do or disagreed with.
Do not commit. Do not invent results. If something fails, report the failure.
