"""Candidate classifiers and the one place a trained model is loaded from a run directory."""
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

CANDIDATES = ('tfidf', 'embeddings', 'finetune')
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_FINETUNE_MODEL = 'answerdotai/ModernBERT-base'


class Embeddings(BaseEstimator, TransformerMixin):
    """Frozen sentence embeddings. Only the model name is pickled; weights load on first use."""

    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != '_encoder'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(self, '_encoder'):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('The embeddings candidate needs the extra: pip install "callswap[embeddings]"') from exc
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder.encode(list(X), show_progress_bar=False)


def tfidf_model(seed=7):
    features = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(max_iter=1000, random_state=seed))


def embeddings_model(seed=7, model_name=DEFAULT_EMBEDDING_MODEL):
    return make_pipeline(Embeddings(model_name), LogisticRegression(max_iter=1000, random_state=seed))


def finetune_model(out, seed=7, base_model=DEFAULT_FINETUNE_MODEL):
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('The fine-tuned candidate needs the extra: pip install "callswap[finetune]"') from exc
    from .finetune import FineTunedClassifier
    return FineTunedClassifier(Path(out) / 'finetuned', base_model, seed)


def load_model(out):
    """Load `model.joblib` from a run directory.

    joblib files are pickles: loading one runs code from it. Only load run directories you created.
    """
    out = Path(out)
    path = out / 'model.joblib'
    if not path.exists():
        raise ValueError(f'No trained model in {out} (expected {path.name}); run `callswap run` first')
    model = joblib.load(path)
    if hasattr(model, 'relocate'):
        model.relocate(out)  # fine-tuned weights live next to model.joblib; the run directory may have moved
    return model
