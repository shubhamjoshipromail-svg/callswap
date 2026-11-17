"""Read a CSV or JSONL log of (text, label) rows and report on its labels."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

MIN_ROWS = 200
RARE_LABEL_ROWS = 20


def normalise(text):
    """Key used to detect duplicates: lower-cased with whitespace collapsed."""
    return ' '.join(text.lower().split())


def _read(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f'Input file not found: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        if path.suffix.lower() == '.csv':
            return list(csv.DictReader(f))
        rows = []
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: not valid JSON ({exc.msg})') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: each JSONL line must be an object')
            rows.append(row)
        return rows


def ingest(path, text_col='text', label_col='label'):
    """Return (records, label_report). Raises ValueError on unusable input."""
    rows = _read(path)
    if rows and (text_col not in rows[0] or label_col not in rows[0]):
        raise ValueError(
            f'Input needs columns "{text_col}" and "{label_col}"; found {sorted(rows[0])}. '
            'Use --text-col / --label-col to point at the right ones.')
    records = []
    for i, row in enumerate(rows):
        text = str(row.get(text_col) or '').strip()
        label = str(row.get(label_col) if row.get(label_col) is not None else '').strip()
        if not text:
            continue
        if not label:
            raise ValueError(f'Row {i + 1} has text but an empty label')
        records.append({
            'id': str(row.get('id') or i), 'text': text, 'label': label, 'norm': normalise(text),
            'group': str(row.get('group') or ''), 'timestamp': str(row.get('timestamp') or ''),
        })
    n_labels = len({r['label'] for r in records})
    if len(records) < MIN_ROWS or n_labels < 2:
        raise ValueError(f'At least {MIN_ROWS} nonempty rows and two labels are required '
                         f'(got {len(records)} rows, {n_labels} labels)')
    counts = Counter(r['label'] for r in records)
    by_text = defaultdict(list)
    for r in records:
        by_text[r['norm']].append(r)
    inconsistent = 0
    for copies in by_text.values():
        labels = Counter(r['label'] for r in copies)
        if len(labels) > 1:
            inconsistent += len(copies)
        # Copies of one text share a split; training uses their majority label.
        majority = sorted(labels, key=lambda x: (-labels[x], x))[0]
        for r in copies:
            r['training_label'] = majority
    return records, {
        'rows': len(records),
        'counts': dict(sorted(counts.items())),
        'rare_labels': sorted(k for k, v in counts.items() if v < RARE_LABEL_ROWS),
        'duplicate_rows': len(records) - len(by_text),
        'teacher_inconsistency_rows': inconsistent,
    }
