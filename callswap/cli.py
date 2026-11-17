"""Command line: `callswap demo | run | serve | label`."""
import argparse
import json
import os
import sys
from importlib import resources
from pathlib import Path

from . import __version__
from .models import CANDIDATES, DEFAULT_FINETUNE_MODEL

RUN_ARTIFACTS = ('train.jsonl', 'calibration.jsonl', 'sealed.jsonl', 'manifest.json', 'model.joblib',
                 'train_result.json', 'calibration_result.json', 'sealed_result.json', 'report.json', 'report.html')
DEMO_NOTE = '''Demo data: the public Banking77 dataset ({rows:,} customer-service queries, 77 intents; PolyAI, CC BY 4.0),
bundled with callswap. Its human labels stand in for "what your LLM answered", so this shows the mechanics,
not agreement with a real LLM. Nothing was sent anywhere and no API key was used.'''


def run(data, out, target_agreement=.99, min_coverage=.70, price_per_1k_calls=.50, monthly_calls=3000000,
        serving_cost=20, embeddings=False, seed=7, text_col='text', label_col='label', finetune=False,
        finetune_model=DEFAULT_FINETUNE_MODEL, only=None, overwrite=False):
    """ingest -> split -> train -> calibrate -> sealed evaluation -> report. Returns the report data."""
    from .calibrate import calibrate
    from .evaluate import evaluate
    from .ingest import ingest
    from .report import report
    from .split import split
    from .train import train
    out = Path(out)
    if not 0 < target_agreement < 1 or not 0 <= min_coverage <= 1:
        raise ValueError('--target-agreement must be between 0 and 1, and --min-coverage between 0 and 1 inclusive')
    if overwrite:
        for name in RUN_ARTIFACTS:
            (out / name).unlink(missing_ok=True)
    elif (out / 'sealed_result.json').exists():
        raise ValueError(f'{out} already contains a sealed evaluation; use a new --out directory')
    records, labels = ingest(data, text_col, label_col)
    split(records, out, seed)
    trained = train(out, seed, embeddings, finetune, finetune_model, only)
    if set(trained['labels']) != set(labels['counts']):
        raise ValueError('Training split cannot preserve every input label under the split and duplicate constraints')
    calibrate(out, target_agreement)
    evaluate(out, target_agreement, min_coverage)
    return report(out, labels, price_per_1k_calls, monthly_calls, serving_cost)


def demo_data():
    return resources.files('callswap') / 'data' / 'demo_banking77.csv'


def _add_run_options(p):
    p.add_argument('--target-agreement', type=float, default=.99, help='required 95%% lower bound on agreement for accepted calls (default 0.99)')
    p.add_argument('--min-coverage', type=float, default=.70, help='minimum share of calls the local model must take over (default 0.70)')
    p.add_argument('--price-per-1k-calls', type=float, default=.50, help='what 1,000 calls to your LLM cost, in dollars (default 0.50)')
    p.add_argument('--monthly-calls', type=int, default=3000000, help='calls per month (default 3,000,000)')
    p.add_argument('--serving-cost', type=float, default=20, help='monthly cost of running the local model, in dollars (default 20)')
    p.add_argument('--json', action='store_true', help='print the headline numbers as JSON instead of text')


def _env_flag(name):
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


def build_parser():
    parser = argparse.ArgumentParser(prog='callswap', description='Replace a repeated LLM classification call with a local model, '
                                     'and get a proof of how often it agrees before you switch.')
    parser.add_argument('--version', action='version', version=f'callswap {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    d = sub.add_parser('demo', help='run the whole pipeline on a small bundled public dataset (no API key, no network)')
    d.add_argument('--out', default='callswap-demo', help='where to write the run (default ./callswap-demo; rewritten each time)')
    _add_run_options(d)

    r = sub.add_parser('run', help='train on your logs and run the sealed test')
    r.add_argument('data', help='CSV or JSONL with columns text,label (optional: id, group, timestamp)')
    r.add_argument('--out', required=True, help='new directory for this run')
    _add_run_options(r)
    r.add_argument('--embeddings', action='store_true', help='also try frozen sentence embeddings (needs callswap[embeddings])')
    r.add_argument('--finetune', action='store_true', help='also try a fine-tuned transformer (needs callswap[finetune])')
    r.add_argument('--finetune-model', default=DEFAULT_FINETUNE_MODEL, help=f'base model to fine-tune (default {DEFAULT_FINETUNE_MODEL})')
    r.add_argument('--only', choices=CANDIDATES, help='use exactly this candidate, with no model selection')
    r.add_argument('--seed', type=int, default=7)
    r.add_argument('--text-col', default='text')
    r.add_argument('--label-col', default='label')

    s = sub.add_parser('serve', help='serve a run as an OpenAI-compatible endpoint with LLM fallback')
    s.add_argument('out', nargs='?', default=os.environ.get('CALLSWAP_RUN_DIR'), help='run directory (or env CALLSWAP_RUN_DIR)')
    s.add_argument('--teacher-url', default=os.environ.get('TEACHER_URL') or None, help='full URL of the LLM chat-completions endpoint to fall back to (or env TEACHER_URL)')
    s.add_argument('--teacher-model', default=os.environ.get('TEACHER_MODEL') or None, help='model name to send to the teacher (or env TEACHER_MODEL); default keeps the caller\'s')
    s.add_argument('--shadow', action='store_true', default=_env_flag('CALLSWAP_SHADOW'), help='always answer from the teacher, and log whether the local model agreed (or env CALLSWAP_SHADOW=1)')
    s.add_argument('--demo-if-missing', action='store_true', help='if the run directory has no model yet, build the bundled demo run there first (used by the Docker image)')
    s.add_argument('--host', default=os.environ.get('CALLSWAP_HOST', '127.0.0.1'))
    s.add_argument('--port', type=int, default=int(os.environ.get('CALLSWAP_PORT', '8000')))

    l = sub.add_parser('label', help='ask an LLM to label a file of texts (creates teacher labels; needs your own key in TEACHER_API_KEY)')
    l.add_argument('data', help='CSV or JSONL with a text column')
    l.add_argument('--out', required=True, help='CSV to write (id,text,label,raw); rerunning resumes')
    l.add_argument('--labels', help='comma-separated list of allowed labels')
    l.add_argument('--labels-file', help='file with one allowed label per line')
    l.add_argument('--teacher-url', default=os.environ.get('TEACHER_URL') or None, required=not os.environ.get('TEACHER_URL'))
    l.add_argument('--teacher-model', default=os.environ.get('TEACHER_MODEL') or None, required=not os.environ.get('TEACHER_MODEL'))
    l.add_argument('--text-col', default='text')
    l.add_argument('--prompt-file', help='system prompt template; {labels} is replaced by the label list')
    l.add_argument('--concurrency', type=int, default=4)
    l.add_argument('--limit', type=int, help='label at most this many new rows')
    return parser


def _summary(result):
    from .report import summary
    return summary(result)


def _print_result(result, as_json):
    from .report import summary
    if as_json:
        ev, econ = result['evaluation'], result['economics']
        print(json.dumps({'verdict': ev['verdict'], 'coverage': ev['coverage'], 'agreement': ev['agreement'],
                          'agreement_lower_bound': ev['agreement_lower_bound'], 'calls_saved_per_1000': econ['calls_saved_per_1000'],
                          'monthly_saving': econ['monthly_saving']}, indent=2))
    else:
        print(summary(result))


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop('command')
    try:
        if command in ('run', 'demo'):
            as_json, out = args.pop('json'), args['out']
            if command == 'demo':
                with resources.as_file(demo_data()) as path:
                    result = run(path, overwrite=True, only='tfidf', **args)
                if not as_json:
                    print(DEMO_NOTE.format(rows=result['label_report']['rows']) + '\n')
            else:
                result = run(**args)
            _print_result(result, as_json)
            if not as_json:
                print(f'\nFull report: {Path(out) / "report.html"}')
                print(f'Serve it:    callswap serve {out}   (then POST to http://127.0.0.1:8000/v1/chat/completions)')
                if command == 'demo':
                    print('Your data:   callswap run your_logs.csv --out runs/first   (CSV or JSONL with columns text,label)')
        elif command == 'serve':
            import uvicorn

            from .serve import create_app
            host, port, out = args.pop('host'), args.pop('port'), args['out']
            if not out:
                raise ValueError('Give a run directory: callswap serve RUN_DIR (or set CALLSWAP_RUN_DIR)')
            if args.pop('demo_if_missing') and not (Path(out) / 'model.joblib').exists():
                print(f'No run in {out} yet: building the bundled Banking77 demo run (about half a minute)...', file=sys.stderr, flush=True)
                with resources.as_file(demo_data()) as path:
                    print(_summary(run(path, out, overwrite=True, only='tfidf')), file=sys.stderr, flush=True)
            app = create_app(**args)
            verdict_path = Path(out) / 'sealed_result.json'
            verdict = json.loads(verdict_path.read_text())['verdict'] if verdict_path.exists() else 'not evaluated'
            if verdict != 'PASS':
                print(f'WARNING: this run\'s sealed verdict is {verdict}. Consider --shadow until a run passes.', file=sys.stderr)
            if not args['teacher_url']:
                print('WARNING: no teacher configured (TEACHER_URL). Low-confidence requests get the local guess, marked '
                      'x-callswap-route: local-unsure.', file=sys.stderr)
            elif not os.environ.get('TEACHER_API_KEY'):
                print('NOTE: TEACHER_API_KEY is not set; the caller\'s Authorization header will be forwarded to the teacher.', file=sys.stderr)
            uvicorn.run(app, host=host, port=port)
        else:
            from .label import DEFAULT_PROMPT, label_file
            if bool(args['labels']) == bool(args['labels_file']):
                raise ValueError('Give exactly one of --labels or --labels-file')
            labels = args['labels'].split(',') if args['labels'] else Path(args['labels_file']).read_text(encoding='utf-8').splitlines()
            prompt = Path(args['prompt_file']).read_text(encoding='utf-8') if args['prompt_file'] else DEFAULT_PROMPT
            counts = label_file(args['data'], args['out'], labels, args['teacher_url'], args['teacher_model'], args['text_col'],
                                prompt, args['concurrency'], args['limit'])
            print(json.dumps(counts, indent=2))
            if counts['unparsed']:
                print(f'{counts["unparsed"]} replies did not name exactly one label; they have an empty label in {args["out"]}. '
                      'Fix or remove them before `callswap run`.', file=sys.stderr)
    except (ValueError, RuntimeError) as exc:
        print(f'callswap: error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
