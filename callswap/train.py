"""Fit the candidate models and pick one. Reads the train split only."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, cross_val_score

from .models import CANDIDATES, DEFAULT_FINETUNE_MODEL, embeddings_model, finetune_model, tfidf_model
from .split import read_rows


def _group_validation_slice(rows, y, seed):
    """Indices for a stratified 90/10 split of train that keeps copies of one text on the same side."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['norm'], []).append(i)
    keys = list(groups)
    group_labels = [y[groups[key][0]] for key in keys]
    fit_groups, val_groups = next(StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(keys, group_labels))
    return [i for g in fit_groups for i in groups[keys[g]]], [i for g in val_groups for i in groups[keys[g]]]


def _finetune_details(model, base_model):
    return {'base_model': base_model, 'device': model.device_, 'epochs_run': model.epochs_run_,
            'max_length': model.max_length_, 'training_seconds': model.training_seconds_}


def train(out, seed=7, embeddings=False, finetune=False, finetune_model_name=DEFAULT_FINETUNE_MODEL, only=None):
    """Train candidates on `train.jsonl`, keep the best, write `model.joblib` and `train_result.json`.

    `only` forces a single candidate (no selection). With several candidates the winner is chosen by
    macro-F1: 5-fold duplicate-aware cross-validation, or a 10% validation slice when fine-tuning
    (a fine-tuned model is too slow to cross-validate).
    """
    out = Path(out)
    rows = read_rows(out / 'train.jsonl')
    x, y = [r['text'] for r in rows], [r['training_label'] for r in rows]
    if only is not None and only not in CANDIDATES:
        raise ValueError(f'Unknown candidate "{only}"; choose from {", ".join(CANDIDATES)}')
    names = [only] if only else ['tfidf'] + ['embeddings'] * bool(embeddings) + ['finetune'] * bool(finetune)
    build = {'tfidf': lambda: tfidf_model(seed), 'embeddings': lambda: embeddings_model(seed),
             'finetune': lambda: finetune_model(out, seed, finetune_model_name)}
    candidates = {name: build[name]() for name in names}

    if len(candidates) == 1:
        winner = names[0]
        model = candidates[winner].fit(x, y)
        result = {'winner': winner, 'selection_method': 'single_candidate'}
    elif 'finetune' in candidates:
        fit_idx, val_idx = _group_validation_slice(rows, y, seed)
        fx, fy = [x[i] for i in fit_idx], [y[i] for i in fit_idx]
        vx, vy = [x[i] for i in val_idx], [y[i] for i in val_idx]
        scores = {}
        for name, candidate in candidates.items():
            if name == 'finetune':
                candidate.fit(fx, fy, validation_data=(vx, vy))
                prediction = candidate.classes_[candidate.decision_function(vx).argmax(axis=1)]
            else:
                prediction = candidate.fit(fx, fy).predict(vx)
            scores[name] = {'macro_f1': float(f1_score(vy, prediction, average='macro', zero_division=0)),
                            'accuracy': float(accuracy_score(vy, prediction))}
        winner = max(scores, key=lambda name: scores[name]['macro_f1'])
        model = candidates[winner]  # kept as fitted on the 90%; the fine-tuned model is not refitted
        result = {'winner': winner, 'selection_method': 'stratified_group_validation', 'validation_scores': scores}
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        groups = [r['norm'] for r in rows]
        scores = {name: float(np.mean(cross_val_score(candidate, x, y, groups=groups, cv=cv, scoring='f1_macro', n_jobs=1)))
                  for name, candidate in candidates.items()}
        winner = max(scores, key=scores.get)
        model = candidates[winner].fit(x, y)
        result = {'winner': winner, 'selection_method': 'cross_validation', 'cv_macro_f1': scores}

    if 'finetune' in candidates:
        result.update(_finetune_details(candidates['finetune'], finetune_model_name))
    result.update({'candidates': names, 'train_rows': len(rows), 'labels': [str(c) for c in model.classes_]})
    joblib.dump(model, out / 'model.joblib')
    (out / 'train_result.json').write_text(json.dumps(result, indent=2))
    return result
