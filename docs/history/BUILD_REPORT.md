# Build report

Implemented CSV/JSONL ingestion, duplicate and group-aware splitting, TF-IDF classification with optional embeddings, calibration, sealed evaluation, economics, HTML/JSON reporting, FastAPI serving, synthetic tests, and a Banking77 demo. Created `.venv` and installed the dependencies (including SciPy for optimization).

## Final test output

Command: `.venv/bin/pytest -q`

```text
....                                                                     [100%]
=============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/fastapi/testclient.py:1
  /Users/shubhamjoshi/dev/callswap/.venv/lib/python3.13/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
4 passed, 1 warning in 1.49s
```

The first pytest attempt failed to import the package; pytest path configuration fixed it. A subsequent attempt failed on a syntax error in the Wilson calculation, and another reported two failing assertions involving chronological duplicate constraints and a threshold test. Those issues were corrected before the final run above.

## Banking77 demo

Command: `.venv/bin/python scripts/demo_banking77.py`

Exact output:

```text
Banking77 download failed: HTTP Error 404: Not Found
```

The demo exited with status 1 on the download failure. No Banking77 data, evaluation, or numbers were produced. The current Hugging Face repository contains Parquet rather than the specified raw CSV files. No alternate dataset or format was substituted. An earlier demo attempt also failed on the Wilson calculation syntax error, which was fixed before this final attempt.

## Limitations and deviations

The real-data demo could not complete because the specified raw CSV URLs returned 404. The optional sentence-transformers path was not installed or exercised. Chronological splitting rejects data when duplicate or group constraints prevent three nonempty chronological splits. Training also rejects a split when its majority-label training data cannot preserve all input labels. The tests use a compact synthetic dataset and do not cover every edge case or independently prove all guarantees for arbitrary inputs.

## Fine-tune addendum

Added an optional transformer fine-tuning candidate, duplicate-aware stratified training validation and candidate selection, lazy-loaded weights, CLI flags, training details in reports, optional dependencies, and tests for the sealed boundary and bert-tiny round-trip. The original cross-validation path remains in use without `--finetune`. No Banking77 fine-tune was run.

Default suite, command `.venv/bin/pytest -q`:

```text
.....                                                                    [100%]
5 passed, 1 deselected in 1.65s
```

Slow test, command `HF_HOME=/private/tmp/callswap-hf .venv/bin/pytest -q -m slow -o addopts='' --tb=short`:

```text
.                                                                        [100%]
1 passed, 5 deselected in 6.70s
```

The first slow-test attempt failed because the default Hugging Face cache was outside the sandbox's writable paths; a writable temporary cache allowed the download. Subsequent attempts exposed tokenizer/config compatibility with the installed Transformers version and a lazy-load bug; these were fixed before the successful run. The requested slow test used CPU; MPS was not tested.
