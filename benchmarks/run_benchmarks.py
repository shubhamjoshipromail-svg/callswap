"""Run every (dataset, model) pair through the sealed pipeline and write benchmarks/results/*.json.

    python benchmarks/run_benchmarks.py                      # everything (fine-tuning takes hours on a laptop)
    python benchmarks/run_benchmarks.py --models tfidf       # the fast baseline only
    python benchmarks/run_benchmarks.py --datasets banking77 --models tfidf embeddings

The labels are the datasets' own human labels standing in for "what the LLM answered". To benchmark against a
real LLM teacher, relabel a dataset with `callswap label` and pass it with --data NAME=PATH.

Each model is trained once per dataset. The calibration threshold is then chosen separately for each agreement
bar (on the calibration split only) and scored on the sealed split, so the sealed split is read once per
(dataset, model, bar) and never influences training, selection or thresholds.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import SOURCES, build  # noqa: E402

from callswap.calibrate import calibrate  # noqa: E402
from callswap.economics import economics  # noqa: E402
from callswap.evaluate import evaluate  # noqa: E402
from callswap.ingest import ingest  # noqa: E402
from callswap.models import CANDIDATES  # noqa: E402
from callswap.split import split  # noqa: E402
from callswap.train import train  # noqa: E402

BARS = (.99, .95, .90)
MIN_COVERAGE = .70
SEED = 7
RESULTS = Path(__file__).resolve().parent / 'results'
SHARED = ('train.jsonl', 'calibration.jsonl', 'sealed.jsonl', 'manifest.json', 'model.joblib', 'train_result.json')


def versions():
    found = {}
    for package in ('callswap', 'scikit-learn', 'numpy', 'scipy', 'sentence-transformers', 'torch', 'transformers'):
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return found


def bench(name, data_path, sha, model_name, runs_dir, teacher):
    base = runs_dir / name / model_name
    base.mkdir(parents=True, exist_ok=True)
    records, label_report = ingest(data_path)
    meta_path = base / 'bench_meta.json'
    if not (base / 'model.joblib').exists() or not meta_path.exists():
        split(records, base, SEED)
        start = time.monotonic()
        train(base, SEED, only=model_name)
        meta_path.write_text(json.dumps({'train_seconds': time.monotonic() - start}))
    train_result = json.loads((base / 'train_result.json').read_text())
    manifest = json.loads((base / 'manifest.json').read_text())
    bars = {}
    for bar in BARS:
        run = base / f'bar_{round(bar * 100)}'
        if run.exists():
            shutil.rmtree(run)
        run.mkdir()
        for item in SHARED:
            shutil.copy2(base / item, run / item)
        if (base / 'finetuned').exists():
            os.symlink(base / 'finetuned', run / 'finetuned')
        cal = calibrate(run, bar)
        ev = evaluate(run, bar, MIN_COVERAGE)
        per = ev['per_label']
        bars[f'{bar:.2f}'] = {
            'verdict': ev['verdict'], 'fail_reasons': ev['fail_reasons'], 'coverage': ev['coverage'],
            'agreement': ev['agreement'], 'agreement_lower_bound': ev['agreement_lower_bound'],
            'accepted': ev['accepted'], 'threshold': cal['threshold'], 'temperature': cal['temperature'],
            'target_met_on_calibration': cal['target_met_on_calibration'],
            'calls_saved_per_1000': economics(ev['coverage'])['calls_saved_per_1000'],
            'monthly_saving_at_defaults': economics(ev['coverage'])['monthly_saving'],
            'weakest_labels': [{'label': l, **{k: per[l][k] for k in ('count', 'accepted', 'accepted_disagreements', 'coverage', 'agreement')}}
                               for l in ev['weakest_labels'][:5]],
            'labels_never_accepted': sum(1 for v in per.values() if v['accepted'] == 0),
        }
        latency, accuracy, sealed_rows, unproven = ev['latency_ms'], ev['overall_accuracy'], ev['sealed_rows'], ev['labels_unproven']
        print(f"  {name:10s} {model_name:10s} bar {bar:.0%}: {ev['verdict']}  coverage {ev['coverage']:.1%}  "
              f"lower bound {ev['agreement_lower_bound']:.2%}", flush=True)
    result = {
        'dataset': name, 'title': SOURCES.get(name, {}).get('title', name), 'model': model_name, 'teacher': teacher,
        'data_sha256': sha, 'rows': label_report['rows'], 'labels': len(label_report['counts']),
        'duplicate_rows': label_report['duplicate_rows'], 'teacher_inconsistency_rows': label_report['teacher_inconsistency_rows'],
        'split': manifest['counts'], 'sealed_rows': sealed_rows, 'seed': SEED, 'min_coverage': MIN_COVERAGE,
        'overall_accuracy': accuracy, 'latency_ms': latency, 'labels_unproven': len(unproven),
        'train_seconds': json.loads(meta_path.read_text())['train_seconds'],
        'training': {k: v for k, v in train_result.items() if k != 'labels'},
        'bars': bars,
        'environment': {'python': platform.python_version(), 'machine': platform.machine(), 'system': platform.system(),
                        'processor': platform.processor(), 'packages': versions()},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'{name}__{model_name}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument('--models', nargs='+', default=list(CANDIDATES), choices=CANDIDATES)
    parser.add_argument('--data', action='append', default=[], metavar='NAME=PATH',
                        help='benchmark your own file (for example teacher labels from `callswap label`) under NAME')
    parser.add_argument('--teacher', default='gold labels (stand-in)', help='how the labels were produced, recorded in the results')
    parser.add_argument('--runs-dir', default=str(ROOT / 'runs' / 'bench'))
    args = parser.parse_args()
    import hashlib
    jobs = [(name, *build(name)[:2]) for name in args.datasets] if not args.data else []
    for item in args.data:
        name, _, path = item.partition('=')
        jobs.append((name, Path(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    for model_name in args.models:  # all datasets on the fast models first
        for name, path, sha in jobs:
            bench(name, path, sha, model_name, Path(args.runs_dir), args.teacher)
    print('Done. Rebuild the table with: python benchmarks/make_table.py')


if __name__ == '__main__':
    main()
