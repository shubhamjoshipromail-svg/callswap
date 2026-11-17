"""Download the public benchmark datasets from their publishers and write data/<name>.csv (text,label).

Nothing here is redistributed with callswap except the small demo subset of Banking77.
Usage: python benchmarks/datasets.py [name ...]
"""
import csv
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'

SOURCES = {
    'banking77': {
        'title': 'Banking77', 'task': 'banking support intents', 'licence': 'CC BY 4.0',
        'publisher': 'https://github.com/PolyAI-LDN/task-specific-datasets',
        'urls': ['https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv',
                 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv'],
    },
    'clinc150': {
        'title': 'CLINC150', 'task': 'assistant intents across 10 domains, plus out-of-scope', 'licence': 'CC BY 3.0',
        'publisher': 'https://github.com/clinc/oos-eval',
        'urls': ['https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json'],
    },
    'massive': {
        'title': 'MASSIVE (en-US)', 'task': 'voice assistant intents', 'licence': 'CC BY 4.0',
        'publisher': 'https://github.com/alexa/massive',
        'urls': ['https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz'],
    },
    'sms_spam': {
        'title': 'SMS Spam Collection', 'task': 'spam or not (2 labels)', 'licence': 'CC BY 4.0',
        'publisher': 'https://archive.ics.uci.edu/dataset/228/sms+spam+collection',
        'urls': ['https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'],
    },
}


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'callswap-benchmarks'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def banking77(urls):
    for url in urls:
        for row in csv.DictReader(io.StringIO(fetch(url).decode('utf-8-sig'))):
            yield row['text'], row['category']


def clinc150(urls):
    data = json.loads(fetch(urls[0]))
    for part in sorted(data):  # train, val, test and their oos_* counterparts
        for text, intent in data[part]:
            yield text, intent


def massive(urls):
    with tarfile.open(fileobj=io.BytesIO(fetch(urls[0])), mode='r:gz') as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith('data/en-US.jsonl'))
        for line in archive.extractfile(member).read().decode('utf-8').splitlines():
            row = json.loads(line)
            yield row['utt'], row['intent']


def sms_spam(urls):
    with zipfile.ZipFile(io.BytesIO(fetch(urls[0]))) as archive:
        for line in archive.read('SMSSpamCollection').decode('utf-8').splitlines():
            label, _, text = line.partition('\t')
            yield text, label


def build(name):
    """Download one dataset (unless already present) and return (path, sha256, rows)."""
    path = DATA / f'{name}.csv'
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        rows = [(' '.join(text.split()), label) for text, label in globals()[name](SOURCES[name]['urls']) if text.strip()]
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            writer.writerows(rows)
    raw = path.read_bytes()
    return path, hashlib.sha256(raw).hexdigest(), raw.count(b'\n') - 1


if __name__ == '__main__':
    for name in sys.argv[1:] or SOURCES:
        path, sha, rows = build(name)
        print(f'{name}: {rows} rows, sha256 {sha[:16]}…, {path}')
