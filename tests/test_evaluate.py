import builtins
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import set_threshold

from callswap.calibrate import calibrate, predict, wilson
from callswap.evaluate import evaluate
from callswap.ingest import ingest
from callswap.models import load_model
from callswap.split import read_rows, split
from callswap.train import train


def forbid_sealed(monkeypatch):
    """Make any attempt to open sealed.jsonl fail, through every route the code could use."""
    originals = {'open': builtins.open, 'path_open': Path.open, 'read_bytes': Path.read_bytes, 'read_text': Path.read_text}

    def check(path):
        if str(path).endswith('sealed.jsonl'):
            raise AssertionError('sealed.jsonl was opened before the sealed evaluation')

    def guard(original):
        def wrapped(path, *args, **kwargs):
            check(path)
            return original(path, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(builtins, 'open', guard(originals['open']))
    monkeypatch.setattr(Path, 'open', guard(originals['path_open']))
    monkeypatch.setattr(Path, 'read_bytes', guard(originals['read_bytes']))
    monkeypatch.setattr(Path, 'read_text', guard(originals['read_text']))


def test_train_and_calibrate_never_open_the_sealed_file(dataset, tmp_path, monkeypatch):
    records, _ = ingest(dataset)
    out = tmp_path / 'run'
    split(records, out)
    with monkeypatch.context() as m:
        forbid_sealed(m)
        train(out)
        calibrate(out, .9)
        with pytest.raises(AssertionError, match='sealed'):
            evaluate(out, .9, .5)  # proves the guard works: evaluate is the one reader
    assert evaluate(out, .9, .5)['verdict'] in ('PASS', 'FAIL')


class FakeFineTuned:
    device_, epochs_run_, max_length_, training_seconds_ = 'cpu', 1, 8, 0.0

    def fit(self, texts, labels, validation_data=None):
        train_norms = {' '.join(t.lower().split()) for t in texts}
        assert not train_norms & {' '.join(t.lower().split()) for t in validation_data[0]}
        self.classes_ = np.array(sorted(set(labels)))
        return self

    def decision_function(self, texts):
        return np.array([[float(t.lower().startswith(c)) for c in self.classes_] for t in texts])


def test_finetune_selection_never_opens_the_sealed_file(dataset, tmp_path, monkeypatch):
    records, _ = ingest(dataset)
    out = tmp_path / 'run'
    split(records, out)

    monkeypatch.setattr('callswap.train.finetune_model', lambda *a, **k: FakeFineTuned())
    with monkeypatch.context() as m:
        forbid_sealed(m)
        result = train(out, finetune=True)
        calibrate(out, .9)
    assert result['selection_method'] == 'stratified_group_validation'
    assert set(result['validation_scores']) == {'tfidf', 'finetune'}
    assert result['base_model'] and result['epochs_run'] == 1


def test_refuses_tampered_sealed_file_and_second_run(dataset, tmp_path):
    records, _ = ingest(dataset)
    out = tmp_path / 'run'
    split(records, out)
    train(out)
    calibrate(out, .9)
    sealed = out / 'sealed.jsonl'
    original = sealed.read_bytes()
    sealed.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate(out, .9, .5)
    lines = original.splitlines()
    sealed.write_bytes(b'\n'.join(lines[:-1]) + b'\n')  # drop a row
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate(out, .9, .5)
    assert not (out / 'sealed_result.json').exists()
    sealed.write_bytes(original)
    evaluate(out, .9, .5)
    with pytest.raises(ValueError, match='already'):
        evaluate(out, .9, .5)


def fresh(run_dir):
    (run_dir / 'sealed_result.json').unlink()
    return run_dir


def test_numbers_are_consistent_with_an_independent_count(run_dir):
    result = evaluate(fresh(run_dir), .9, .5)
    cal = json.loads((run_dir / 'calibration_result.json').read_text())
    rows = read_rows(run_dir / 'sealed.jsonl')
    pred, conf = predict(load_model(run_dir), [r['text'] for r in rows], cal['temperature'])
    truth = np.array([r['label'] for r in rows])
    accepted = conf >= cal['threshold']
    agreed = int((accepted & (pred == truth)).sum())
    assert result['sealed_rows'] == len(rows) and result['accepted'] == int(accepted.sum())
    assert result['coverage'] == pytest.approx(accepted.mean())
    assert result['agreement'] == pytest.approx(agreed / accepted.sum())
    assert result['agreement_lower_bound'] == pytest.approx(wilson(agreed, int(accepted.sum())))
    assert result['overall_accuracy'] == pytest.approx((pred == truth).mean())
    per = result['per_label']
    assert sum(v['count'] for v in per.values()) == len(rows)
    assert sum(v['accepted'] for v in per.values()) == result['accepted']
    assert sum(v['accepted_disagreements'] for v in per.values()) == result['accepted'] - agreed
    assert len(result['disagreements']) == min(20, result['accepted'] - agreed)
    assert result['latency_ms']['p50'] > 0


def test_abstention_threshold_controls_what_is_accepted(run_dir):
    set_threshold(fresh(run_dir), 2.0)  # nothing can be this confident
    nothing = evaluate(run_dir, .9, .5)
    assert nothing['accepted'] == 0 and nothing['coverage'] == 0 and nothing['agreement'] is None
    assert nothing['verdict'] == 'FAIL' and len(nothing['fail_reasons']) == 2
    set_threshold(fresh(run_dir), 0.0)  # accept everything
    everything = evaluate(run_dir, .9, .5)
    assert everything['coverage'] == 1 and everything['agreement'] == pytest.approx(everything['overall_accuracy'])


def test_verdict_needs_both_coverage_and_bound(run_dir):
    set_threshold(fresh(run_dir), 0.0)
    assert evaluate(run_dir, .5, .5)['verdict'] == 'PASS'
    by_bound = evaluate(fresh(run_dir), .9999, .5)
    assert by_bound['verdict'] == 'FAIL' and 'lower bound' in by_bound['fail_reasons'][0]
    set_threshold(fresh(run_dir), .999999)
    by_coverage = evaluate(run_dir, .0001, .99)
    assert by_coverage['verdict'] == 'FAIL' and any('coverage' in r for r in by_coverage['fail_reasons'])


def test_small_labels_are_reported_as_unproven(run_dir):
    result = evaluate(fresh(run_dir), .9, .5)
    assert result['labels_unproven'] == sorted(k for k, v in result['per_label'].items() if v['count'] < 20)
    assert result['labels_missing_from_sealed'] == []
    assert set(result['weakest_labels']) <= set(result['per_label'])
