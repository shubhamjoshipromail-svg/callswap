import json

import pytest
from conftest import make_rows, write_csv

from callswap.ingest import ingest, normalise


def test_csv_report_counts(dataset):
    records, report = ingest(dataset)
    assert report['rows'] == len(records) == 452
    assert report['counts'] == {'alpha': 156, 'beta': 156, 'gamma': 140}
    assert report['duplicate_rows'] == 2
    assert report['rare_labels'] == [] and report['teacher_inconsistency_rows'] == 0
    assert records[0]['norm'] == normalise(records[0]['text'])


def test_jsonl_and_blank_lines(tmp_path, rows):
    path = tmp_path / 'input.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n\n', encoding='utf-8')
    assert ingest(path)[1]['rows'] == 452


def test_bom_and_custom_columns(tmp_path, rows):
    renamed = [{'utterance': r['text'], 'intent': r['label']} for r in rows]
    path = write_csv(tmp_path / 'input.csv', renamed)
    path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
    assert ingest(path, 'utterance', 'intent')[1]['rows'] == 452
    with pytest.raises(ValueError, match='columns'):
        ingest(path)


def test_optional_columns_are_kept(tmp_path, rows):
    for i, r in enumerate(rows):
        r.update(id=f'row-{i}', group=f'g{i % 50}', timestamp=f'2026-01-{1 + i % 28:02}')
    records, _ = ingest(write_csv(tmp_path / 'input.csv', rows))
    assert records[3]['id'] == 'row-3' and records[3]['group'] == 'g3' and records[3]['timestamp'] == '2026-01-04'


def test_empty_text_dropped_and_empty_label_rejected(tmp_path, rows):
    path = write_csv(tmp_path / 'a.csv', rows + [{'text': '   ', 'label': 'alpha'}])
    assert ingest(path)[1]['rows'] == 452
    with pytest.raises(ValueError, match='empty label'):
        ingest(write_csv(tmp_path / 'b.csv', rows + [{'text': 'something', 'label': ' '}]))


@pytest.mark.parametrize('bad, message', [
    (make_rows()[:150], 'At least 200'),
    ([{'text': f'only one label {i}', 'label': 'x'} for i in range(250)], 'two labels'),
])
def test_too_small_inputs_are_refused(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message):
        ingest(write_csv(tmp_path / 'input.csv', bad))


def test_missing_file_and_bad_json(tmp_path):
    with pytest.raises(ValueError, match='not found'):
        ingest(tmp_path / 'nope.csv')
    bad = tmp_path / 'bad.jsonl'
    bad.write_text('{"text": "a", "label": "b"}\nnot json\n')
    with pytest.raises(ValueError, match='bad.jsonl:2'):
        ingest(bad)
    bad.write_text('[1, 2]\n')
    with pytest.raises(ValueError, match='must be an object'):
        ingest(bad)


def test_conflicting_duplicates_use_majority_label(tmp_path, rows):
    rows += [{'text': 'Refund my order', 'label': 'alpha'}, {'text': 'refund my  order', 'label': 'alpha'},
             {'text': 'REFUND MY ORDER', 'label': 'beta'}]
    records, report = ingest(write_csv(tmp_path / 'input.csv', rows))
    copies = [r for r in records if r['norm'] == 'refund my order']
    assert report['teacher_inconsistency_rows'] == 3
    assert {r['training_label'] for r in copies} == {'alpha'}
    assert sorted(r['label'] for r in copies) == ['alpha', 'alpha', 'beta']  # original labels are preserved


def test_rare_labels_are_flagged_not_dropped(tmp_path, rows):
    rows += [{'text': f'rare thing {i}', 'label': 'rare'} for i in range(5)]
    _, report = ingest(write_csv(tmp_path / 'input.csv', rows))
    assert report['rare_labels'] == ['rare'] and report['counts']['rare'] == 5
