"""Produce teacher labels by asking an OpenAI-compatible chat endpoint to classify each text.

Use this when you do not already have logs of what your LLM answered. The key is read from the
TEACHER_API_KEY environment variable and is never written to disk.
"""
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .ingest import _read

DEFAULT_PROMPT = ('Classify the user message into exactly one of these labels:\n{labels}\n'
                  'Reply with the label only, exactly as written above, and nothing else.')


def match_label(reply, labels):
    """Map a model reply to one of the allowed labels, or None if it does not name exactly one."""
    cleaned = reply.strip().strip('"\'`.').strip()
    by_lower = {label.lower(): label for label in labels}
    if cleaned.lower() in by_lower:
        return by_lower[cleaned.lower()]
    mentioned = [label for label in labels if label.lower() in cleaned.lower()]
    # A longer label can contain a shorter one ("card" in "card_lost"); keep only the longest matches.
    mentioned = [l for l in mentioned if not any(l != other and l.lower() in other.lower() for other in mentioned)]
    return mentioned[0] if len(mentioned) == 1 else None


def label_file(data, out, labels, teacher_url, teacher_model, text_col='text', prompt=DEFAULT_PROMPT, concurrency=4,
               limit=None, transport=None, timeout=60.0, retries=3):
    """Write `out` (CSV: id, text, label, raw) with the teacher's label for each row of `data`.

    Resumable: rows whose id is already in `out` are skipped. Replies that do not name exactly one label
    are written with an empty label so you can inspect `raw`; `callswap run` skips nothing silently, so
    filter or fix those rows before training.
    """
    rows = _read(data)
    if rows and text_col not in rows[0]:
        raise ValueError(f'Input needs a "{text_col}" column; found {sorted(rows[0])}')
    labels = [l.strip() for l in labels if l.strip()]
    if len(set(labels)) < 2:
        raise ValueError('At least two distinct labels are required')
    items = [(str(row.get('id') or i), str(row[text_col]).strip()) for i, row in enumerate(rows) if str(row.get(text_col) or '').strip()]
    out = Path(out)
    done = set()
    if out.exists():
        with out.open(encoding='utf-8', newline='') as f:
            done = {row['id'] for row in csv.DictReader(f)}
    todo = [item for item in items if item[0] not in done][:limit]
    key = os.environ.get('TEACHER_API_KEY')
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    system = prompt.format(labels='\n'.join(f'- {l}' for l in labels))
    client = httpx.Client(transport=transport, timeout=timeout, headers=headers)

    def ask(item):
        payload = {'model': teacher_model, 'temperature': 0,
                   'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': item[1]}]}
        error = None
        for _ in range(retries):
            try:
                response = client.post(teacher_url, json=payload)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404):
                    break  # retrying will not help
        raise RuntimeError(f'Teacher request failed for row {item[0]}: {type(error).__name__}: {error}')

    counts = {'labelled': 0, 'unparsed': 0, 'skipped_existing': len(items) - len([i for i in items if i[0] not in done])}
    new_file = not out.exists()
    try:
        with out.open('a', encoding='utf-8', newline='') as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
            writer = csv.DictWriter(f, fieldnames=['id', 'text', 'label', 'raw'])
            if new_file:
                writer.writeheader()
            for item, reply in zip(todo, pool.map(ask, todo)):
                label = match_label(reply, labels)
                counts['labelled' if label else 'unparsed'] += 1
                writer.writerow({'id': item[0], 'text': item[1], 'label': label or '', 'raw': reply.strip()[:500]})
                f.flush()
                done_now = counts['labelled'] + counts['unparsed']
                if done_now % 100 == 0:
                    print(f'  {done_now}/{len(todo)} labelled', file=sys.stderr, flush=True)
    finally:
        client.close()
    return counts
