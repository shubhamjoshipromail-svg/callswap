"""Optional fine-tuned transformer classifier. Weights live in a directory; the pickle holds only settings."""
import math
import random
import time
from pathlib import Path

import numpy as np

from .ingest import normalise

BATCH_SIZE = 32


def pick_device(preferred=None):
    import torch
    available = {'mps': torch.backends.mps.is_available(), 'cuda': torch.cuda.is_available(), 'cpu': True}
    if preferred and available.get(preferred):
        return preferred
    return next(name for name in ('mps', 'cuda', 'cpu') if available[name])


class FineTunedClassifier:
    def __init__(self, model_path, base_model='answerdotai/ModernBERT-base', seed=7, epochs=6, device=None):
        self.model_path = str(Path(model_path).resolve())
        self.base_model, self.seed, self.epochs, self.device = base_model, seed, epochs, device

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ('_model', '_tokenizer')}

    def relocate(self, run_dir):
        """Point at the weights inside `run_dir` (the run may have been moved or mounted elsewhere)."""
        self.model_path = str((Path(run_dir) / Path(self.model_path).name).resolve())

    def _load(self):
        import torch
        if not hasattr(self, '_model'):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # The training device may not exist where the model is served (e.g. trained on MPS, served on Linux).
            self.device_ = pick_device(self.device or getattr(self, 'device_', None))
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device_).eval()
        return torch

    def _validation_slice(self, texts, labels):
        """Stratified 10% of the given data, keeping copies of one text on the same side."""
        from sklearn.model_selection import StratifiedShuffleSplit
        groups = {}
        for i, text in enumerate(texts):
            groups.setdefault(normalise(text), []).append(i)
        keys = list(groups)
        group_labels = [labels[groups[key][0]] for key in keys]
        fit_groups, val_groups = next(
            StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=self.seed).split(keys, group_labels))
        fit_idx = [i for g in fit_groups for i in groups[keys[g]]]
        val_idx = [i for g in val_groups for i in groups[keys[g]]]
        take = lambda items, idx: [items[i] for i in idx]  # noqa: E731
        return take(texts, fit_idx), take(labels, fit_idx), (take(texts, val_idx), take(labels, val_idx))

    def fit(self, texts, labels, validation_data=None):
        import torch
        from sklearn.metrics import f1_score
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        texts, labels = list(texts), list(labels)
        self.classes_ = np.array(sorted(set(labels)))
        if validation_data is None:
            texts, labels, validation_data = self._validation_slice(texts, labels)
        start = time.monotonic()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device_ = pick_device(self.device)
        print(f'Fine-tuning {self.base_model} on device: {self.device_}', flush=True)
        lookup = {label: i for i, label in enumerate(self.classes_)}
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        except (OSError, ValueError):
            # bert-tiny publishes BERT weights without its tokenizer files.
            if self.base_model != 'prajjwal1/bert-tiny':
                raise
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
        self.max_length_ = min(256, max(8, int(math.ceil(np.percentile(lengths, 95) / 8) * 8)))
        if self.base_model == 'prajjwal1/bert-tiny':
            from transformers import BertConfig
            config = BertConfig.from_pretrained(self.base_model, num_labels=len(self.classes_))
            model = AutoModelForSequenceClassification.from_pretrained(self.base_model, config=config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=len(self.classes_))
        model.to(self.device_)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length_, return_tensors='pt')
        targets = torch.tensor([lookup[y] for y in labels])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*encoded.values(), targets), batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
        steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * .1), steps)
        path = Path(self.model_path)
        path.mkdir(parents=True, exist_ok=True)
        self._model, self._tokenizer = model, tokenizer
        val_texts, val_labels = validation_data
        best, stale = -1, 0
        for epoch in range(self.epochs):
            model.train()
            for batch in loader:
                inputs = {key: value.to(self.device_) for key, value in zip(encoded.keys(), batch[:-1])}
                loss = model(**inputs, labels=batch[-1].to(self.device_)).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            predictions = self.classes_[self.decision_function(val_texts).argmax(axis=1)]
            score = f1_score(val_labels, predictions, average='macro', zero_division=0)
            self.epochs_run_ = epoch + 1
            print(f'  epoch {epoch + 1}: validation macro-F1 {score:.4f}', flush=True)
            if score > best:
                best, stale = score, 0
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
            else:
                stale += 1
                if stale >= 2:
                    break
        # Drop the in-memory model so the next use reloads the best saved epoch.
        del self._model, self._tokenizer
        self.best_validation_macro_f1_ = float(best)
        self.training_seconds_ = time.monotonic() - start
        return self

    def decision_function(self, texts):
        torch = self._load()
        texts, outputs = list(texts), []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                inputs = self._tokenizer(texts[start:start + BATCH_SIZE], padding=True, truncation=True,
                                         max_length=self.max_length_, return_tensors='pt')
                logits = self._model(**{k: v.to(self.device_) for k, v in inputs.items()}).logits
                outputs.append(logits.float().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, len(self.classes_)))
