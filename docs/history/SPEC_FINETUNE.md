# callswap: fine-tuned candidate (addendum to SPEC.md)

v0 result on Banking77 at the 99% agreement bar: TF-IDF takes over 40% of calls, frozen embeddings 33%. Bar is 70%. Both FAIL.
Goal: add a fine-tuned small encoder as a third candidate. Everything in SPEC.md's non-negotiable rules still holds.

## What to build
1. `callswap/finetune.py`: class `FineTunedClassifier` with the same surface the rest of the code already uses:
   `fit(texts, labels)`, `decision_function(texts) -> np.ndarray [n, n_classes]` (raw logits), `classes_` (sorted label array),
   and it must survive `joblib.dump/load` (store weights in a directory inside the run dir, e.g. `runs/NAME/finetuned/`; the
   pickled object holds only the path, label list and settings, and lazy-loads the model). For two classes still return two columns.
2. Training details (fixed, do not tune further):
   - base model from `--finetune-model`, default `answerdotai/ModernBERT-base`
   - device: MPS if available, else CUDA, else CPU; print which one is used
   - max length = min(256, 95th percentile token length of the training texts, rounded up to a multiple of 8)
   - batch size 32, AdamW, lr 5e-5, weight decay 0.01, linear warmup 10% then linear decay, up to 6 epochs
   - early stopping on macro-F1 of the **validation slice** (see 3) with patience 2; restore the best epoch
   - seeds fixed from `--seed` (python, numpy, torch)
   - plain PyTorch training loop or `transformers.Trainer`, whichever is shorter; no extra frameworks
3. Model selection when `--finetune` is passed: carve a stratified 10% **validation slice out of train** (deterministic from the seed,
   duplicates kept together using the existing `norm` field). Fit every candidate (tfidf; embeddings if `--embeddings`; finetune) on
   the remaining 90% and compare macro-F1 on the validation slice. The winner is kept **as fitted on the 90%** (do not refit the
   fine-tuned model). Without `--finetune` the existing 5-fold CV behaviour is unchanged.
   Record per-candidate validation macro-F1 and accuracy, the selection method, base model, device, epochs run, max length and
   training seconds in `train_result.json`, and show them in the report.
4. The calibration and sealed splits are never read by any of this. Extend the existing sealed-boundary test so it still proves that.
5. CLI: `--finetune`, `--finetune-model NAME`. `serve` must work with a fine-tuned run (it already calls `logits(model, texts)`).
6. Dependencies: add `torch` and `transformers` as an optional extra; base tests must still pass without downloading anything.
7. Tests: one test marked `@pytest.mark.slow` (deselected by default in pytest.ini) that fine-tunes `prajjwal1/bert-tiny` for 1 epoch
   on the synthetic dataset on CPU, round-trips through joblib, and checks `decision_function` shape and `classes_`.
   Run that slow test once yourself and report its output.

## Do NOT
- Do not run the Banking77 fine-tune. The architect runs it outside your sandbox (GPU access). Only run unit tests and the bert-tiny slow test.
- Do not change the acceptance bar, the split logic, calibration or evaluation maths.
- Do not commit.

## Report
Append a section "Fine-tune addendum" to BUILD_REPORT.md: what changed, exact test output (default suite and the slow test),
and anything you could not do. If the bert-tiny download or MPS is blocked in your sandbox, say so plainly.
