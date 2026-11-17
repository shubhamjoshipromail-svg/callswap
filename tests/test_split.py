import json
from collections import Counter

import pytest
from conftest import write_csv

from callswap.ingest import ingest
from callswap.split import SPLITS, digest, read_rows, sealed_ids_hash, split


def parts_of(out):
    return {name: read_rows(out / f'{name}.jsonl') for name in SPLITS}


def test_duplicates_never_straddle_splits(dataset, tmp_path):
    records, _ = ingest(dataset)
    split(records, tmp_path / 's')
    norms = {name: {r['norm'] for r in rows} for name, rows in parts_of(tmp_path / 's').items()}
    assert not norms['train'] & norms['calibration']
    assert not norms['train'] & norms['sealed']
    assert not norms['calibration'] & norms['sealed']
    assert sum(len(v) for v in parts_of(tmp_path / 's').values()) == len(records)


def test_proportions_and_stratification(dataset, tmp_path):
    records, _ = ingest(dataset)
    manifest = split(records, tmp_path / 's')
    total = len(records)
    assert manifest['method'] == 'stratified'
    assert abs(manifest['counts']['train'] / total - .6) < .03
    assert abs(manifest['counts']['calibration'] / total - .2) < .03
    assert abs(manifest['counts']['sealed'] / total - .2) < .03
    for rows in parts_of(tmp_path / 's').values():
        counts = Counter(r['label'] for r in rows)
        assert set(counts) == {'alpha', 'beta', 'gamma'}
        assert max(counts.values()) / min(counts.values()) < 1.3


def test_groups_stay_together_transitively(tmp_path, rows):
    for i, r in enumerate(rows):
        r['group'] = f'customer-{i // 4}'
    # Same text under two different groups links those groups.
    rows.append({'text': rows[0]['text'], 'label': rows[0]['label'], 'group': 'customer-90'})
    records, _ = ingest(write_csv(tmp_path / 'input.csv', rows))
    split(records, tmp_path / 's')
    where = {}
    for name, part in parts_of(tmp_path / 's').items():
        for r in part:
            where.setdefault(r['group'], set()).add(name)
    assert all(len(v) == 1 for v in where.values())
    assert where['customer-0'] == where['customer-90']


def test_time_split_puts_latest_rows_in_sealed(tmp_path, rows):
    import random
    random.Random(1).shuffle(rows)
    for i, r in enumerate(rows):
        r['timestamp'] = f'2026-01-01T00:{i // 60:02}:{i % 60:02}'
    records, _ = ingest(write_csv(tmp_path / 'input.csv', rows))
    manifest = split(records, tmp_path / 's')
    parts = parts_of(tmp_path / 's')
    assert manifest['method'] == 'chronological'
    assert max(r['timestamp'] for r in parts['train']) < min(r['timestamp'] for r in parts['calibration'])
    assert max(r['timestamp'] for r in parts['calibration']) < min(r['timestamp'] for r in parts['sealed'])


def test_time_split_needs_every_timestamp(dataset, tmp_path):
    records, _ = ingest(dataset)
    records[0]['timestamp'] = '2026-01-01'
    with pytest.raises(ValueError, match='timestamps'):
        split(records, tmp_path / 's')


def test_deterministic_per_seed(dataset, tmp_path):
    records, _ = ingest(dataset)
    a, b, c = split(records, tmp_path / 'a', 7), split(records, tmp_path / 'b', 7), split(records, tmp_path / 'c', 8)
    assert a == b
    assert a['sealed_ids_labels_sha256'] != c['sealed_ids_labels_sha256']


def test_manifest_hashes_match_files(dataset, tmp_path):
    records, _ = ingest(dataset)
    out = tmp_path / 's'
    manifest = split(records, out)
    assert manifest == json.loads((out / 'manifest.json').read_text())
    for name in SPLITS:
        assert manifest['split_sha256'][name] == digest(out / f'{name}.jsonl')
    sealed = read_rows(out / 'sealed.jsonl')
    assert manifest['sealed_ids_labels_sha256'] == sealed_ids_hash(sealed) == sealed_ids_hash(sealed[::-1])
    sealed[0]['label'] = 'changed'
    assert sealed_ids_hash(sealed) != manifest['sealed_ids_labels_sha256']


def test_impossible_split_is_refused(dataset, tmp_path):
    records, _ = ingest(dataset)
    for r in records:
        r['group'] = 'everyone'
    with pytest.raises(ValueError, match='nonempty'):
        split(records, tmp_path / 's')
