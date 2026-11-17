import csv
import json

import httpx
import pytest
from conftest import write_csv

from callswap.cli import main
from callswap.label import label_file, match_label

LABELS = ['card', 'card_lost', 'refund']


@pytest.mark.parametrize('reply, expected', [
    ('refund', 'refund'), (' Refund. ', 'refund'), ('"card_lost"', 'card_lost'), ('card', 'card'),
    ('The label is card_lost', 'card_lost'), ('I think refund fits best', 'refund'),
    ('card or refund', None), ('no idea', None), ('', None),
])
def test_match_label(reply, expected):
    assert match_label(reply, LABELS) == expected


def answering(seen, status=200):
    def handler(request):
        seen.append(request)
        if status != 200:
            return httpx.Response(status, json={'error': 'nope'})
        text = json.loads(request.content)['messages'][-1]['content']
        reply = 'refund' if 'money' in text else 'card' if 'card' in text else 'something else entirely'
        return httpx.Response(200, json={'choices': [{'message': {'content': reply}}]})
    return httpx.MockTransport(handler)


@pytest.fixture
def texts(tmp_path):
    rows = [{'id': 'a', 'text': 'money back please'}, {'id': 'b', 'text': 'my card broke'}, {'id': 'c', 'text': 'hello'}]
    return write_csv(tmp_path / 'texts.csv', rows)


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_label_file_writes_labels_and_resumes(texts, tmp_path, monkeypatch):
    monkeypatch.setenv('TEACHER_API_KEY', 'my-key')
    seen, out = [], tmp_path / 'labelled.csv'
    counts = label_file(texts, out, LABELS, 'https://t.test/v1', 'm', transport=answering(seen), limit=2)
    assert counts == {'labelled': 2, 'unparsed': 0, 'skipped_existing': 0}
    payload = json.loads(seen[0].content)
    assert payload['model'] == 'm' and payload['temperature'] == 0 and '- card_lost' in payload['messages'][0]['content']
    assert seen[0].headers['authorization'] == 'Bearer my-key'
    counts = label_file(texts, out, LABELS, 'https://t.test/v1', 'm', transport=answering(seen))
    assert counts == {'labelled': 0, 'unparsed': 1, 'skipped_existing': 2} and len(seen) == 3
    rows = read(out)
    assert [(r['id'], r['label']) for r in rows] == [('a', 'refund'), ('b', 'card'), ('c', '')]
    assert rows[2]['raw'] == 'something else entirely'
    assert 'my-key' not in out.read_text()


def test_label_file_errors(texts, tmp_path, monkeypatch):
    monkeypatch.delenv('TEACHER_API_KEY', raising=False)
    seen = []
    with pytest.raises(RuntimeError, match='row a'):
        label_file(texts, tmp_path / 'o.csv', LABELS, 'https://t.test/v1', 'm', transport=answering(seen, 401))
    assert len(seen) == 3 and 'authorization' not in seen[0].headers  # one attempt per row: 401 is not retried
    with pytest.raises(ValueError, match='two distinct labels'):
        label_file(texts, tmp_path / 'o2.csv', ['one', 'one'], 'https://t.test/v1', 'm')
    with pytest.raises(ValueError, match='column'):
        label_file(texts, tmp_path / 'o3.csv', LABELS, 'https://t.test/v1', 'm', text_col='utterance')


def test_cli_run_end_to_end(dataset, tmp_path, capsys):
    out = tmp_path / 'run'
    assert main(['run', str(dataset), '--out', str(out), '--target-agreement', '0.9', '--min-coverage', '0.5']) == 0
    printed = capsys.readouterr().out
    assert 'callswap verdict:' in printed and 'coverage' in printed and 'calls saved' in printed
    report = json.loads((out / 'report.json').read_text())
    assert report['evaluation']['verdict'] in ('PASS', 'FAIL') and report['label_report']['rows'] == 452
    html = (out / 'report.html').read_text()
    assert 'Verdict' in html and 'not correctness' in html and 'http' not in html.replace('http-equiv', '')
    # A second run into the same directory must not re-use the sealed set.
    assert main(['run', str(dataset), '--out', str(out)]) == 2
    assert 'already contains a sealed evaluation' in capsys.readouterr().err


def test_cli_json_output_and_custom_columns(tmp_path, rows, capsys):
    data = write_csv(tmp_path / 'd.csv', [{'q': r['text'], 'intent': r['label']} for r in rows])
    code = main(['run', str(data), '--out', str(tmp_path / 'r'), '--text-col', 'q', '--label-col', 'intent', '--json',
                 '--target-agreement', '0.9', '--monthly-calls', '1000000'])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and set(result) == {'verdict', 'coverage', 'agreement', 'agreement_lower_bound', 'calls_saved_per_1000', 'monthly_saving'}
    assert result['calls_saved_per_1000'] == pytest.approx(1000 * result['coverage'])


def test_cli_errors_return_2(tmp_path, dataset, capsys, monkeypatch):
    monkeypatch.delenv('CALLSWAP_RUN_DIR', raising=False)
    assert main(['run', str(tmp_path / 'missing.csv'), '--out', str(tmp_path / 'r')]) == 2
    assert main(['run', str(dataset), '--out', str(tmp_path / 'r2'), '--target-agreement', '1.5']) == 2
    assert main(['serve']) == 2
    assert main(['serve', str(tmp_path / 'empty')]) == 2
    assert main(['label', str(dataset), '--out', str(tmp_path / 'l.csv'), '--teacher-url', 'http://x', '--teacher-model', 'm']) == 2
    assert 'callswap: error:' in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(['--version'])
    assert exit_info.value.code == 0 and capsys.readouterr().out.startswith('callswap ')


def test_bundled_demo_data_is_valid():
    from importlib import resources

    from callswap.cli import demo_data
    from callswap.ingest import ingest
    with resources.as_file(demo_data()) as path:
        _, report = ingest(path)
    assert report['rows'] == 13083 and len(report['counts']) == 77


@pytest.mark.slow
def test_demo_command_runs_twice(tmp_path, capsys):
    out = tmp_path / 'demo'
    assert main(['demo', '--out', str(out)]) == 0
    assert main(['demo', '--out', str(out), '--json']) == 0  # demo may overwrite its own directory
    assert (out / 'report.html').exists()


@pytest.mark.slow
def test_finetune_bert_tiny_roundtrip_after_moving_the_run(dataset, tmp_path):
    from callswap.cli import run
    from callswap.models import load_model
    out = tmp_path / 'run'
    report = run(dataset, out, target_agreement=.9, min_coverage=.5, only='finetune', finetune_model='prajjwal1/bert-tiny')
    assert report['training']['winner'] == 'finetune' and report['training']['epochs_run'] >= 1
    moved = tmp_path / 'moved'
    out.rename(moved)
    model = load_model(moved)
    assert list(model.classes_) == ['alpha', 'beta', 'gamma']
    assert model.decision_function(['alpha example', 'beta example']).shape == (2, 3)
