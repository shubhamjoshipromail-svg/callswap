"""The one sealed evaluation. This is the only module that opens `sealed.jsonl`."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .calibrate import predict, wilson
from .models import load_model
from .split import sealed_ids_hash

UNPROVEN_SEALED_ROWS = 20
LATENCY_SAMPLES = 100


def _latency_ms(model, texts, temperature):
    """Median and 95th percentile time to classify one text, measured on this machine."""
    times = []
    for text in texts[:LATENCY_SAMPLES]:
        start = time.perf_counter()
        predict(model, [text], temperature)
        times.append((time.perf_counter() - start) * 1000)
    return {'p50': float(np.percentile(times, 50)), 'p95': float(np.percentile(times, 95)), 'samples': len(times)}


def evaluate(out, target=.99, min_coverage=.70):
    """Score the locked model, temperature and threshold on the sealed split, once.

    Refuses to run if the sealed file no longer matches the manifest or a result already exists.
    """
    out = Path(out)
    result_path = out / 'sealed_result.json'
    if result_path.exists():
        raise ValueError('Sealed evaluation has already been performed for this run directory')
    manifest = json.loads((out / 'manifest.json').read_text())
    raw = (out / 'sealed.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['split_sha256']['sealed']:
        raise ValueError('Sealed file hash mismatch: sealed.jsonl changed after the split')
    rows = [json.loads(line) for line in raw.splitlines()]
    if sealed_ids_hash(rows) != manifest['sealed_ids_labels_sha256']:
        raise ValueError('Sealed ids and labels hash mismatch')
    model = load_model(out)
    cal = json.loads((out / 'calibration_result.json').read_text())
    texts = [r['text'] for r in rows]
    pred, conf = predict(model, texts, cal['temperature'])
    truth = np.array([r['label'] for r in rows])
    accepted, correct = conf >= cal['threshold'], pred == truth
    n, agreed = int(accepted.sum()), int((accepted & correct).sum())
    coverage, bound = n / len(rows), wilson(agreed, n)

    counts = Counter(truth.tolist())
    per_label = {}
    for label in sorted(counts):
        mask = truth == label
        taken = mask & accepted
        k, ok = int(taken.sum()), int((taken & correct).sum())
        per_label[label] = {'count': int(mask.sum()), 'accepted': k, 'accepted_disagreements': k - ok,
                            'coverage': k / int(mask.sum()), 'agreement': ok / k if k else None,
                            'agreement_lower_bound': wilson(ok, k) if k else None}
    # Labels the local model handles worst: most accepted wrong answers first, then lowest coverage.
    weak = sorted(per_label, key=lambda l: (-per_label[l]['accepted_disagreements'], per_label[l]['coverage'], l))
    mistakes = [{'text': rows[i]['text'][:200], 'teacher_label': str(truth[i]), 'prediction': str(pred[i]),
                 'confidence': float(conf[i])} for i in np.argsort(conf) if accepted[i] and not correct[i]][:20]
    reasons = []
    if coverage < min_coverage:
        reasons.append(f'coverage {coverage:.1%} is below the minimum {min_coverage:.0%}')
    if bound < target:
        reasons.append(f'agreement lower bound {bound:.2%} is below the target {target:.0%}')
    result = {
        'verdict': 'FAIL' if reasons else 'PASS',
        'fail_reasons': reasons,
        'target_agreement': target,
        'min_coverage': min_coverage,
        'sealed_rows': len(rows),
        'accepted': n,
        'coverage': coverage,
        'agreement': agreed / n if n else None,
        'agreement_lower_bound': bound,
        'overall_accuracy': float(correct.mean()),
        'latency_ms': _latency_ms(model, texts, cal['temperature']),
        'per_label': per_label,
        'weakest_labels': weak[:10],
        'labels_unproven': sorted(k for k, v in counts.items() if v < UNPROVEN_SEALED_ROWS),
        'labels_missing_from_sealed': sorted(set(map(str, model.classes_)) - set(counts)),
        'disagreements': mistakes,
    }
    # Exclusive creation: a second evaluation of the same run cannot overwrite the first.
    with result_path.open('x') as f:
        json.dump(result, f, indent=2)
    return result
