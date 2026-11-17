# Contributing

Thanks for looking. The most useful contributions right now are **results on real classification calls** (PASS or FAIL: open a
"Share a result" issue) and bug reports with a small reproducing file.

## Setup

```bash
git clone https://github.com/shubhamjoshipromail-svg/callswap
cd callswap
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                 # fast suite, no network, no model downloads
ruff check .
```

The slow tests fine-tune a tiny model and run the bundled demo; they need `pip install -e ".[finetune]"` and a network connection
the first time:

```bash
pytest -m slow -o addopts=''
```

## Rules the code must keep

These are what make the verdict worth something. A change that weakens one needs a very good reason.

1. **Only `callswap/evaluate.py` opens `sealed.jsonl`.** `tests/test_evaluate.py` fails if training or calibration touches it.
2. **The sealed evaluation runs once per run directory** and refuses a file that no longer matches the manifest hash.
3. **No leakage:** duplicates (after lower-casing and collapsing whitespace) and `group` values stay in one split.
4. **Labels are preserved.** Never merge, rename or drop a user's labels; report rare or inconsistent ones instead.
5. **Report honestly.** FAIL is a valid output. Do not add retries, seed searches or defaults that make a pass more likely.
6. **No request text or labels in logs.**
7. **No API keys in the repository**, in tests, in the Docker image or in examples. Users bring their own through
   `TEACHER_API_KEY`.

## Benchmarks

`python benchmarks/run_benchmarks.py` reruns everything (the fine-tuned models take hours on a laptop);
`python benchmarks/make_table.py` rebuilds `benchmarks/RESULTS.md` and the README table from `benchmarks/results/*.json`.
If you change anything that affects numbers, rerun the affected rows and commit the JSON with the regenerated tables. CI checks
that the tables match the JSON.

## Pull requests

Keep them small, add a test for behaviour you change, and say in the description what you ran.
