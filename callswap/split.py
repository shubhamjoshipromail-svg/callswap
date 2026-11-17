"""Split records into train / calibration / sealed without leakage, and write a hashed manifest."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPLITS = ('train', 'calibration', 'sealed')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed_ids_hash(rows):
    pairs = sorted((r['id'], r['label']) for r in rows)
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _components(records):
    """Group rows that must stay together: same normalised text or same `group`, transitively."""
    parent = list(range(len(records)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for i, r in enumerate(records):
        keys = [('text', r['norm'])] + ([('group', r['group'])] if r['group'] else [])
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, r in enumerate(records):
        components[root(i)].append(r)
    return list(components.values())


def split(records, out, seed=7):
    """60% train / 20% calibration / 20% sealed. By time if timestamps exist, else stratified by label."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    comps = _components(records)
    parts = {name: [] for name in SPLITS}
    by_time = any(r['timestamp'] for r in records)
    if by_time:
        if not all(r['timestamp'] for r in records):
            raise ValueError('All rows require timestamps for a time split')
        comps.sort(key=lambda c: (min(r['timestamp'] for r in c), min(r['id'] for r in c)))
        # Merge components whose time ranges overlap, so nothing in a later split predates an earlier one.
        blocks = []
        for c in comps:
            start = min(r['timestamp'] for r in c)
            if blocks and start <= max(r['timestamp'] for r in blocks[-1]):
                blocks[-1].extend(c)
            else:
                blocks.append(list(c))
        if len(blocks) < 3:
            raise ValueError('Time and duplicate/group constraints prevent three chronological splits')
        n = len(records)
        for c in blocks:
            placed = sum(map(len, parts.values()))
            name = 'train' if placed < .6 * n else 'calibration' if placed < .8 * n else 'sealed'
            parts[name].extend(c)
    else:
        np.random.RandomState(seed).shuffle(comps)
        # Assign components label by label (majority label of the component) to keep class proportions.
        by_label = defaultdict(list)
        for c in comps:
            counts = Counter(r['training_label'] for r in c)
            by_label[sorted(counts, key=lambda k: (-counts[k], k))[0]].append(c)
        for label in sorted(by_label):
            total = sum(map(len, by_label[label]))
            placed = 0
            for c in by_label[label]:
                name = 'train' if placed < .6 * total else 'calibration' if placed < .8 * total else 'sealed'
                parts[name].extend(c)
                placed += len(c)
    if any(not rows for rows in parts.values()):
        raise ValueError('Constraints prevent three nonempty splits')
    for name, rows in parts.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    manifest = {
        'seed': seed,
        'method': 'chronological' if by_time else 'stratified',
        'counts': {k: len(v) for k, v in parts.items()},
        'sealed_ids_labels_sha256': sealed_ids_hash(parts['sealed']),
        'split_sha256': {k: digest(out / f'{k}.jsonl') for k in parts},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]
