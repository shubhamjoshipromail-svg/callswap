# Security

## Reporting a problem

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/shubhamjoshipromail-svg/callswap/security/advisories/new),
not as a public issue. You should get a reply within a week.

## Things to know before you deploy

- **Run directories are trusted input.** `model.joblib` is a Python pickle; loading one executes code from it. Only serve or
  evaluate run directories that you produced with `callswap run`. Never load one from an untrusted source.
- **The server has no authentication.** `callswap serve` binds to `127.0.0.1` by default. The Docker image listens on all
  interfaces inside the container, and the provided compose file publishes it on `127.0.0.1` only. If you expose it further, put
  it behind something that authenticates callers, because in fallback mode it spends your LLM budget on their behalf.
- **Keys.** The teacher key is read from the `TEACHER_API_KEY` environment variable at request time. It is not written to disk
  or to logs. If it is unset, the caller's `Authorization` header is forwarded to the teacher URL you configured, and nowhere else.
- **What is logged.** `requests.log` holds a timestamp, the route, the confidence, the latency and the upstream status. It never
  holds the request text or the predicted label. `report.html`, `report.json` and the split files in the run directory do
  contain your input text; treat the run directory like the data it was built from.
- **Fallback sends data to your LLM provider.** Requests the local model is unsure about are forwarded, body unchanged, to the
  `TEACHER_URL` you set. That is the same data your application was already sending there.

## Supported versions

Only the latest release receives fixes.
