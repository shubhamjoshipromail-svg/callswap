# callswap

**Replace a repeated LLM classification call with a local model, and get a proof of how often it agrees before you switch.**

If your product calls an LLM thousands of times a day to answer the same kind of question ("which queue does this ticket go to?",
"which intent is this?", "is this spam?"), most of those calls are easy. callswap trains a small local classifier on the answers
your LLM already gave, lets it answer only when it is confident, and sends everything else to the LLM as before.

Before you switch anything, it runs one sealed test and tells you:

- **coverage**: what share of calls the local model would take over,
- **agreement**: how often it gives the same answer as your LLM on those calls, with a 95% lower confidence bound,
- a **PASS / FAIL verdict** against the bar you set, and where it is weak, label by label.

It reports FAIL when it cannot prove the bar, which on hard datasets is often (see [Results](#results)).

```
callswap verdict: FAIL   (bar: agreement lower bound >= 99%, coverage >= 70%)
  why: coverage 39.6% is below the minimum 70%
  coverage              39.6%   the local model answered 1,025 of 2,586 sealed calls; the rest go to the LLM
  agreement             99.90%  on the calls it answered (95% lower bound 99.45%)
  without abstaining    88.6%   agreement if the local model answered everything
  calls saved           396 of every 1,000
  estimated saving      $575/month at 3,000,000 calls/month, $0.50 per 1,000 calls, $20/month serving
  local latency         1.1 ms median per call on this machine (tfidf model)
```

## 60-second demo

No API key, no account, no network access after install. Needs Python 3.11 to 3.13.

```bash
uvx --from git+https://github.com/shubhamjoshipromail-svg/callswap callswap demo
```

or with pipx / pip:

```bash
pipx run --spec git+https://github.com/shubhamjoshipromail-svg/callswap callswap demo
```

```bash
pip install git+https://github.com/shubhamjoshipromail-svg/callswap && callswap demo
```

The demo runs the whole pipeline on the bundled public Banking77 dataset (13,083 customer-service queries, 77 intents) in about
half a minute and prints the block above. Its human labels stand in for "what your LLM answered", so it shows the mechanics, not
agreement with a real LLM. It writes `callswap-demo/report.html`, and you can serve the result straight away:

```bash
callswap serve callswap-demo
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"any","messages":[{"role":"user","content":"I lost my card, please block it"}]}'
```

## Use it on your own logs

You need a CSV or JSONL file with two columns: `text` (what you sent to the LLM) and `label` (what it answered).
At least 200 rows; a few thousand is realistic for a 99% bar, because proving 99% needs at least 381 answered calls in the sealed
split even with zero disagreements.

```csv
text,label
"My card still hasn't arrived",card_arrival
"Why was I charged twice?",duplicate_charge
```

```bash
callswap run your_logs.csv --out runs/first
```

Optional columns make the test stricter and more realistic:

| Column | Effect |
|---|---|
| `timestamp` (ISO 8601) | Splits by time: trains on the oldest 60%, tests on the newest 20%. Use this if you have it. |
| `group` | Rows with the same group (a customer, a conversation) never straddle train and test. |
| `id` | Kept in the split files so you can trace rows. |

Useful options (`callswap run --help` lists all):

| Option | Default | Meaning |
|---|---|---|
| `--target-agreement` | `0.99` | Required 95% lower bound on agreement for the calls the local model answers. |
| `--min-coverage` | `0.70` | Minimum share of calls it must take over for a PASS. |
| `--embeddings` | off | Also try sentence embeddings + logistic regression. Needs `pip install "callswap[embeddings]"`. |
| `--finetune` | off | Also try a fine-tuned ModernBERT. Needs `pip install "callswap[finetune]"`; minutes to an hour on a GPU or Apple Silicon. |
| `--only tfidf\|embeddings\|finetune` | | Use exactly one model, with no selection. |
| `--price-per-1k-calls`, `--monthly-calls`, `--serving-cost` | `0.50`, `3000000`, `20` | Inputs to the savings estimate. |
| `--text-col`, `--label-col` | `text`, `label` | Column names in your file. |

Do not have logged answers yet? `callswap label` asks any OpenAI-compatible endpoint to label a file of texts, using **your own key**
from the `TEACHER_API_KEY` environment variable (it is never written to disk):

```bash
export TEACHER_API_KEY=...   # your key
callswap label texts.csv --out labelled.csv --labels-file labels.txt --teacher-url https://api.openai.com/v1/chat/completions --teacher-model YOUR_MODEL
```

## What it does, and why you can trust the number

1. **Ingest.** Drops empty rows, finds duplicates (ignoring case and spacing) and counts texts your LLM labelled inconsistently.
   Your labels are never merged, renamed or dropped.
2. **Split** into 60% train, 20% calibration, 20% **sealed**. Duplicates and groups always land in the same split. The sealed
   split is hashed into `manifest.json` before any training happens.
3. **Train** candidates on the train split only and pick one by duplicate-aware cross-validation.
4. **Calibrate** on the calibration split only: fit a temperature, then choose the lowest confidence threshold at which the 95%
   Wilson lower bound of agreement reaches your target. Below that threshold the model abstains.
5. **Sealed test, once.** Only now is the sealed split opened. The code refuses to run it twice for a run directory, and refuses
   if the file no longer matches its hash. A test in this repository fails if training or calibration ever opens the sealed file.
6. **Report**: `report.html` and `report.json` with the verdict, the per-label table and the lowest-confidence disagreements.

Because the threshold is fixed before the sealed split is read, the sealed number is not flattered by threshold tuning. That is
the difference between "accuracy on a validation set I also tuned on" and a test result.

### Reading the verdict

- **PASS** means: on unseen data from the same distribution, the local model answered at least `--min-coverage` of calls, and with
  95% confidence it agrees with your LLM on at least `--target-agreement` of the calls it answers.
- **FAIL on coverage only** (the agreement bound is met) means the local answers are proven to your bar, but it takes fewer calls
  than you asked for. You can still route that share locally; the saving is just smaller.
- **FAIL on the bound** means it is not proven. The threshold is chosen right at the edge on calibration data, so a sealed bound
  slightly under the bar is common. Do not rerun with new seeds until it passes: that is how a sealed test stops being one.
  Get more data, try `--embeddings` or `--finetune`, or accept a lower bar in a fresh run and say so.
- The "95% lower bound" is the lower end of a two-sided 95% Wilson interval (equivalently a one-sided 97.5% bound).

## Deploy the server

`callswap serve RUN_DIR` exposes an OpenAI-compatible `POST /v1/chat/completions` (non-streaming). Point your existing client's
base URL at it. For each request it classifies the last user message, and:

| Situation | What the caller gets | `x-callswap-route` header |
|---|---|---|
| Confident | The label as the assistant message, in about a millisecond | `local` |
| Not confident, teacher configured | Your LLM's reply, passed through unchanged | `teacher` |
| Not confident, no teacher configured | The local guess, flagged | `local-unsure` |
| Teacher unreachable or returned an error | HTTP 502, or the teacher's own error status | `teacher-error` |
| Shadow mode | Always your LLM's reply; callswap logs whether it would have agreed | `teacher` |

Also: `POST /classify` (`{"text": ...}` → label, confidence, accepted), `GET /stats` (counts by route, shadow agreement rate),
`GET /healthz`. One line per request goes to `RUN_DIR/requests.log`: route, confidence, latency. **Never the text or the label.**

The model was trained on the `text` column, so send exactly that as the user message. If your prompt wraps the text in a template,
train on the wrapped text.

### Docker

```bash
git clone https://github.com/shubhamjoshipromail-svg/callswap && cd callswap && docker compose up
```

That serves the bundled demo run on `http://127.0.0.1:8000` (the first start trains it, about half a minute). To configure the LLM
fallback, copy `.env.example` to `.env` and fill it in. Everything is set through environment variables; no key is ever stored
in the image or the repository.

| Variable | Meaning |
|---|---|
| `TEACHER_URL` | Full URL of the chat-completions endpoint to fall back to. Empty means no fallback. |
| `TEACHER_MODEL` | Model name to send to the teacher. Empty keeps the caller's. |
| `TEACHER_API_KEY` | Your key for the teacher. If empty, the caller's own `Authorization` header is forwarded. |
| `CALLSWAP_SHADOW` | `1` answers everything from the teacher and only logs agreement. Start here in production. |
| `CALLSWAP_RUN_DIR` | Run directory inside the container (default `/runs/demo`). |

Train on your own logs inside the container (put the file in `./data`), then point `CALLSWAP_RUN_DIR` at the result:

```bash
docker compose run --rm callswap run /data/your_logs.csv --out /runs/mine
```

A sensible rollout: run `--shadow` for a week, check `/stats` for the live agreement rate, then turn shadow off.

## Results

Four public datasets, three models, one sealed test each. The datasets' human labels stand in for the LLM, because no LLM key
was available when these were produced; `benchmarks/run_benchmarks.py --data NAME=PATH` reruns the same benchmark on labels from
a real LLM (create them with `callswap label`).

<!-- results:start -->
Teacher labels: **gold labels (stand-in)**. Bar: 95% Wilson lower bound on agreement of at least 99%, and at least 70% of calls answered locally. 3 of 12 dataset and model pairs pass.

| Dataset | Model | Verdict | Coverage | Agreement on answered | 95% lower bound | Answering everything | Latency (median) | Training |
|---|---|---|---|---|---|---|---|---|
| Banking77 | TF-IDF + logistic regression | **FAIL** | 39.6% | 99.90% | 99.45% | 88.6% | 1.3 ms | 15 s |
| Banking77 | MiniLM embeddings + logistic regression | **FAIL** | 32.7% | 99.88% | 99.33% | 90.2% | 7.5 ms | 13 s |
| Banking77 | Fine-tuned ModernBERT-base | **FAIL** | 39.7% | 99.81% | 99.29% | 92.1% | 20.8 ms | 42 min |
| CLINC150 | TF-IDF + logistic regression | **FAIL** | 56.0% | 99.40% | 99.02% | 90.0% | 0.9 ms | 60 s |
| CLINC150 | MiniLM embeddings + logistic regression | **PASS** | 75.8% | 99.42% | 99.11% | 93.8% | 7.5 ms | 10 s |
| CLINC150 | Fine-tuned ModernBERT-base | **FAIL** | 90.6% | 99.16% | 98.84% | 96.0% | 19.7 ms | 38 min |
| MASSIVE (en-US) | TF-IDF + logistic regression | **FAIL** | 22.9% | 99.47% | 98.64% | 83.4% | 0.8 ms | 27 s |
| MASSIVE (en-US) | MiniLM embeddings + logistic regression | **FAIL** | 11.7% | 99.74% | 98.54% | 84.5% | 7.8 ms | 72 s |
| MASSIVE (en-US) | Fine-tuned ModernBERT-base | **FAIL** | 28.7% | 99.58% | 98.91% | 88.4% | 20.4 ms | 26 min |
| SMS Spam Collection | TF-IDF + logistic regression | **FAIL** | 69.0% | 99.87% | 99.27% | 97.2% | 1.0 ms | 1 s |
| SMS Spam Collection | MiniLM embeddings + logistic regression | **PASS** | 82.6% | 99.67% | 99.05% | 97.3% | 9.1 ms | 5 s |
| SMS Spam Collection | Fine-tuned ModernBERT-base | **PASS** | 87.3% | 99.90% | 99.42% | 99.2% | 20.2 ms | 17 min |
<!-- results:end -->

Coverage at 95% and 90% bars, per-label weak spots and dataset details are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
Rebuild the tables from the committed result files with `python benchmarks/make_table.py`; rerun everything from scratch with
`python benchmarks/run_benchmarks.py`.

## Limitations

- **Agreement is not correctness.** Every number is agreement with the labels you supplied. If your LLM was wrong, the local
  model learns to be wrong the same way, and callswap will report high agreement.
- **Rare labels are unproven.** A PASS is about calls overall. Labels with fewer than 20 sealed examples are listed as unproven;
  read the per-label table before trusting a label that matters.
- **Drift is not measured.** The result holds for inputs like the ones in your file. Use a `timestamp` column, keep shadow mode on
  for a while, and re-run when your traffic or label set changes. There is no automatic drift detection or retraining.
- **Single-label classification only.** No multi-label, extraction, scoring or free-text generation. No streaming responses.
- **The benchmark teachers are human labels**, not an LLM, and each benchmark is one split with one seed.
- **Not a security boundary.** A run directory contains a pickled model; only serve runs you trained. The server has no
  authentication of its own: bind it to localhost or put it behind your gateway. See [SECURITY.md](SECURITY.md).

## How it compares

Checked against each project's own page on 16 November 2026. All of them train a small model on an LLM's answers; they differ
in what they prove and how they fit into an existing app. If one of these fits your case better, use it.

| | callswap | [TRACER](https://github.com/adrida/tracer) | [Jevstiller](https://github.com/tomerglick57/Jevstiller) | [LLMCostCut](https://github.com/zhaoliangvaio/llmcostcut) | [Expressible distill](https://github.com/expressibleai/expressible-cli) |
|---|---|---|---|---|---|
| Input | Raw text + labels (CSV/JSONL) | Traces plus embeddings you supply | Live traffic to the Jev API | Live calls through its `monitor()` | 50+ labelled examples |
| Local model | TF-IDF, embeddings or fine-tuned transformer | Classical ML on your embeddings | Frozen encoder + logistic regression | DistilBERT encoder + small head | MiniLM embeddings + small network |
| Abstains and defers to the LLM | Yes | Yes | Yes | Yes (fixed confidence threshold, default 0.8) | Not described |
| Test set kept apart from training and threshold choice | Yes, sealed, hashed, run once | Yes, reserved certification rows | Calibration set also picks the threshold; continuous audit slice | Not described | Validation split for early stopping |
| Confidence bound | Wilson, 95% | Clopper–Pearson, one-sided 90% by default | Clopper–Pearson, 95% | None described | None described |
| Duplicate, group and time-aware split | Yes, built in | No; its README warns these can invalidate the bound | Not described | Not described | Not described |
| Drop-in endpoint | OpenAI-compatible, with fallback and shadow mode | HTTP server that takes embeddings | Proxy for the Jev API, with shadow testing | Python API | CLI and exported JS |
| Drift detection, continual retraining | **No** | Manual `update()` | **Yes**, both | Online learning | No |
| Runs without any API key | Yes (bundled demo) | Needs traces and embeddings | Needs a Jev key | Needs a teacher | Yes |

Where callswap is behind: Jevstiller detects drift and retrains continuously, and TRACER has a larger community, more model
families and an out-of-distribution guard. Where callswap is different: it starts from a plain CSV of text and labels, guards the
split against duplicates, groups and time leakage, and puts an OpenAI-compatible endpoint with fallback in front of the result.

## Development

```bash
git clone https://github.com/shubhamjoshipromail-svg/callswap && cd callswap && pip install -e ".[dev]" && pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The original build specifications are kept in [docs/history](docs/history).

## Licence

MIT, see [LICENSE](LICENSE). The bundled demo data is Banking77 by PolyAI, CC BY 4.0 (see [callswap/data/NOTICE.md](callswap/data/NOTICE.md)).
