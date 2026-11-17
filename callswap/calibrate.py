"""Temperature scaling and the abstain threshold. Reads the calibration split only."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from .models import load_model
from .split import read_rows

Z95 = 1.959963984540054
CURVE_POINTS = 200


def wilson(successes, n):
    """Lower end of the 95% Wilson score interval for a proportion (0.0 when n is 0)."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    safe = np.where(n > 0, n, 1.0)
    p = successes / safe
    bound = (p + Z95**2 / (2 * safe) - Z95 * np.sqrt(p * (1 - p) / safe + Z95**2 / (4 * safe**2))) / (1 + Z95**2 / safe)
    bound = np.where(n > 0, np.maximum(bound, 0.0), 0.0)
    return float(bound) if bound.ndim == 0 else bound


def logits(model, texts):
    """Class scores as an [n, n_classes] array, for binary models too."""
    scores = np.asarray(model.decision_function(texts))
    if scores.ndim == 1:
        scores = np.column_stack((-scores / 2, scores / 2))
    return scores


def predict(model, texts, temperature):
    """Return (predicted labels, confidence) using temperature-scaled probabilities."""
    probs = softmax(logits(model, texts) / temperature, axis=1)
    return model.classes_[probs.argmax(axis=1)], probs.max(axis=1)


def choose_threshold(conf, correct, target):
    """Pick the lowest confidence threshold whose accepted set has a Wilson lower bound >= target.

    Every distinct confidence value is a candidate, so every attainable accepted set is considered.
    If no threshold reaches the target, return the one with the best lower bound (ties: more coverage).
    Returns (threshold, target_met, curve) where curve is a down-sampled list of points for plotting.
    """
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    order = np.argsort(-conf, kind='stable')
    sorted_conf, hits = conf[order], np.cumsum(correct[order])
    # Last index of each run of equal confidences: accepting conf >= value takes all of them.
    last = np.flatnonzero(np.append(sorted_conf[1:] != sorted_conf[:-1], True))
    thresholds, accepted, agreed = sorted_conf[last], last + 1, hits[last]
    bounds = wilson(agreed, accepted)
    passing = np.flatnonzero(bounds >= target)
    if len(passing):
        pick, met = passing[-1], True  # thresholds are descending, so the last passing one is the lowest
    else:
        pick, met = np.lexsort((accepted, bounds))[-1], False
    keep = np.unique(np.append(np.linspace(0, len(last) - 1, min(len(last), CURVE_POINTS)).astype(int), pick))
    curve = [{'threshold': float(thresholds[i]), 'coverage': float(accepted[i] / len(conf)),
              'agreement': float(agreed[i] / accepted[i]), 'lower_bound': float(bounds[i]),
              'accepted': int(accepted[i])} for i in keep]
    return float(thresholds[pick]), met, curve


def fit_temperature(scores, label_index):
    """Single scalar that minimises negative log-likelihood of the labels (rows with unseen labels are skipped)."""
    known = label_index >= 0
    if not known.any():
        return 1.0
    scores, label_index = scores[known], label_index[known]

    def nll(log_t):
        return -log_softmax(scores / np.exp(log_t), axis=1)[np.arange(len(scores)), label_index].mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method='bounded').x))


def calibrate(out, target=.99):
    out = Path(out)
    model = load_model(out)
    rows = read_rows(out / 'calibration.jsonl')
    labels = np.array([r['label'] for r in rows])
    scores = logits(model, [r['text'] for r in rows])
    index = {label: i for i, label in enumerate(model.classes_)}
    temperature = fit_temperature(scores, np.array([index.get(y, -1) for y in labels]))
    probs = softmax(scores / temperature, axis=1)
    threshold, met, curve = choose_threshold(probs.max(axis=1), model.classes_[probs.argmax(axis=1)] == labels, target)
    result = {'target_agreement': target, 'temperature': temperature, 'threshold': threshold,
              'target_met_on_calibration': met, 'calibration_rows': len(rows), 'curve': curve}
    (out / 'calibration_result.json').write_text(json.dumps(result, indent=2))
    return result
