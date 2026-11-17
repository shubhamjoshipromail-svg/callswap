import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from callswap.economics import economics
from callswap.ingest import ingest
from callswap.models import Embeddings, load_model
from callswap.split import split
from callswap.train import train


@pytest.fixture
def split_dir(dataset, tmp_path):
    records, _ = ingest(dataset)
    out = tmp_path / 'run'
    split(records, out)
    return out


def test_single_candidate_keeps_every_label(split_dir):
    result = train(split_dir, only='tfidf')
    assert result['winner'] == 'tfidf' and result['selection_method'] == 'single_candidate'
    assert result['labels'] == ['alpha', 'beta', 'gamma']
    assert json.loads((split_dir / 'train_result.json').read_text()) == result
    model = load_model(split_dir)
    assert model.predict(['gamma distinctive topic sample number 999'])[0] == 'gamma'


def test_default_is_tfidf_alone(split_dir):
    assert train(split_dir)['candidates'] == ['tfidf']


def test_cross_validation_picks_the_better_candidate(split_dir, monkeypatch):
    weak = lambda seed=7: make_pipeline(CountVectorizer(), DummyClassifier(strategy='most_frequent'))  # noqa: E731
    monkeypatch.setattr('callswap.train.embeddings_model', weak)
    result = train(split_dir, embeddings=True)
    assert result['selection_method'] == 'cross_validation' and result['winner'] == 'tfidf'
    assert result['cv_macro_f1']['tfidf'] > result['cv_macro_f1']['embeddings']
    strong = lambda seed=7: make_pipeline(CountVectorizer(), LogisticRegression(max_iter=500))  # noqa: E731
    monkeypatch.setattr('callswap.train.embeddings_model', strong)
    monkeypatch.setattr('callswap.train.tfidf_model', weak)
    assert train(split_dir, embeddings=True)['winner'] == 'embeddings'


def test_unknown_candidate_and_missing_model(split_dir, tmp_path):
    with pytest.raises(ValueError, match='Unknown candidate'):
        train(split_dir, only='magic')
    with pytest.raises(ValueError, match='No trained model'):
        load_model(tmp_path)


def test_embeddings_pickle_holds_no_weights():
    emb = Embeddings('some/model')
    emb._encoder = object()
    assert '_encoder' not in emb.__getstate__() and emb.__getstate__()['model_name'] == 'some/model'


def test_economics_hand_computed():
    e = economics(.5, 1, 100000, 20)
    assert e['baseline'] == 100 and e['after'] == 70 and e['monthly_saving'] == 30
    assert e['percent_saving'] == 30 and e['break_even_monthly_calls'] == 40000
    assert e['calls_saved_per_1000'] == 500 and e['monthly_calls_saved'] == 50000


def test_economics_no_saving_cases():
    assert economics(.1, 1, 1000, 20)['monthly_saving'] < 0
    zero = economics(0, .5, 3000000, 20)
    assert zero['monthly_saving'] == -20 and zero['break_even_monthly_calls'] is None
    assert economics(.5, .5, 0, 20)['percent_saving'] == 0
    assert np.isclose(economics(.4, .5, 3000000, 20)['monthly_saving'], 580)
