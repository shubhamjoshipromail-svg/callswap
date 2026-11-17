import csv
import json

import pytest

from callswap.cli import run


def make_rows(per_label=140, labels=('alpha', 'beta', 'gamma')):
    """Separable labels, a shared ambiguous region (so abstaining matters) and near-duplicate rows."""
    rows = []
    for label in labels:
        for i in range(per_label):
            rows.append({'text': f'{label} distinctive topic sample number {i}', 'label': label})
    for i in range(30):  # ambiguous: same wording, labels alternate
        rows.append({'text': f'unclear generic request variant {i}', 'label': labels[i % 2]})
    rows.append({'text': '  ALPHA distinctive   topic sample number 1 ', 'label': 'alpha'})  # duplicate up to case/space
    rows.append({'text': 'Beta Distinctive Topic Sample Number 2', 'label': 'beta'})
    return rows


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def rows():
    return make_rows()


@pytest.fixture
def dataset(tmp_path, rows):
    return write_csv(tmp_path / 'input.csv', rows)


@pytest.fixture(scope='session')
def trained_run(tmp_path_factory):
    """One full pipeline run shared by the tests that only read it."""
    root = tmp_path_factory.mktemp('shared')
    data = write_csv(root / 'input.csv', make_rows())
    out = root / 'run'
    report = run(data, out, target_agreement=.9, min_coverage=.5, only='tfidf')
    return out, report


@pytest.fixture
def run_dir(trained_run, tmp_path):
    """A private copy of the shared run, safe to modify."""
    import shutil
    out = tmp_path / 'run'
    shutil.copytree(trained_run[0], out)
    return out


def set_threshold(out, value):
    path = out / 'calibration_result.json'
    cal = json.loads(path.read_text())
    cal['threshold'] = value
    path.write_text(json.dumps(cal))
