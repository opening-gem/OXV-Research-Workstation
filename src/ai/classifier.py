"""
Research Classifier for NASA Combustion Investigations.

Uses TF-IDF vectorization + Logistic Regression for multi-label
classification of investigations into research categories.
Designed for small datasets (15-50 investigations).
Built to run with either native NumPy engine or scikit-learn.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import MultiLabelBinarizer
    SKLEARN_AVAILABLE = True
except ImportError:
    from src.ai.compat_ml import (
        TfidfVectorizer,
        OneVsRestLogisticRegression,
        MultiLabelBinarizer,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResearchClassifier:
    """
    Classifies NASA combustion research investigations into research categories.

    Uses TF-IDF vectorization + Logistic Regression as baseline.
    Designed to work with small datasets (15-50 investigations).
    """

    CATEGORIES: list[str] = [
        "spacecraft_fire_safety",
        "flame_spread",
        "material_flammability",
        "droplet_combustion",
        "cool_flames",
        "soot_formation",
        "gas_jet_flames",
        "electric_field_effects",
        "smoke_detection",
        "flame_extinguishment",
        "fire_suppression",
        "solid_fuel_combustion",
        "ignition_dynamics",
    ]

    def __init__(self, model_path: Path | None = None) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        if SKLEARN_AVAILABLE:
            self.classifier = OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                )
            )
        else:
            self.classifier = OneVsRestLogisticRegression(C=1.0, max_iter=300)

        self.label_binarizer = MultiLabelBinarizer()
        self.is_trained: bool = False
        self._feature_names: list[str] = []

        if model_path is not None:
            self.load(model_path)

    def prepare_training_data(
        self, investigations: list[dict[str, Any]]
    ) -> tuple[list[str], np.ndarray]:
        """Convert investigations to training format."""
        if not investigations:
            raise ValueError("investigations list must not be empty")

        texts: list[str] = []
        labels: list[list[str]] = []

        for inv in investigations:
            if "text" not in inv or "categories" not in inv:
                raise ValueError(
                    f"Investigation missing required keys 'text'/'categories': "
                    f"{inv.get('investigation_id', '<unknown>')}"
                )
            texts.append(inv["text"])
            labels.append(inv["categories"])

        y = self.label_binarizer.fit_transform(labels)
        return texts, y

    def train(self, investigations: list[dict[str, Any]]) -> dict[str, Any]:
        """Train the classifier and return training metrics."""
        texts, y = self.prepare_training_data(investigations)
        n_samples = len(texts)

        X = self.vectorizer.fit_transform(texts)
        if hasattr(self.vectorizer, "get_feature_names_out"):
            self._feature_names = list(self.vectorizer.get_feature_names_out())
        else:
            self._feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        n_folds = min(5, n_samples)
        if n_samples < 6:
            n_folds = n_samples
        elif n_samples < 20:
            n_folds = 3

        cv_metrics = self._cross_validate(X, y, n_folds)

        self.classifier.fit(X, y)
        self.is_trained = True
        logger.info(
            "Classifier trained on %d samples with %d categories",
            n_samples,
            y.shape[1],
        )
        return cv_metrics

    def _cross_validate(
        self, X: Any, y: np.ndarray, n_folds: int
    ) -> dict[str, Any]:
        """Run cross-validation and compute metrics."""
        n_samples, n_labels = y.shape
        per_label_pred = np.zeros_like(y)

        if SKLEARN_AVAILABLE:
            for label_idx in range(n_labels):
                y_col = y[:, label_idx]
                pos_cnt = int(y_col.sum())
                neg_cnt = n_samples - pos_cnt
                if pos_cnt < 1 or neg_cnt < 1:
                    per_label_pred[:, label_idx] = 0
                    continue
                max_cv = min(n_folds, pos_cnt, neg_cnt)
                if max_cv < 2:
                    # Rare category with 1 sample cannot support >=2 fold cross validation
                    per_label_pred[:, label_idx] = 0
                    continue
                try:
                    preds = cross_val_predict(
                        LogisticRegression(
                            C=1.0,
                            max_iter=1000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                        X,
                        y_col,
                        cv=max_cv,
                    )
                    per_label_pred[:, label_idx] = preds
                except Exception:
                    per_label_pred[:, label_idx] = 0
        else:
            # Native k-fold evaluation
            indices = np.arange(n_samples)
            np.random.seed(42)
            shuffled = np.random.permutation(indices)
            fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
            fold_sizes[: n_samples % n_folds] += 1
            current = 0
            for fold_size in fold_sizes:
                val_idx = shuffled[current : current + fold_size]
                train_idx = np.setdiff1d(shuffled, val_idx)
                current += fold_size

                X_train, y_train = X[train_idx], y[train_idx]
                X_val = X[val_idx]
                fold_model = OneVsRestLogisticRegression(C=1.0, max_iter=200)
                fold_model.fit(X_train, y_train)
                per_label_pred[val_idx] = fold_model.predict(X_val)

        categories = list(self.label_binarizer.classes_)
        per_category: dict[str, dict[str, float]] = {}
        for idx, cat in enumerate(categories):
            y_true_col = y[:, idx]
            y_pred_col = per_label_pred[:, idx]
            
            tp = int(np.sum((y_true_col == 1) & (y_pred_col == 1)))
            fp = int(np.sum((y_true_col == 0) & (y_pred_col == 1)))
            fn = int(np.sum((y_true_col == 1) & (y_pred_col == 0)))
            support = int(np.sum(y_true_col == 1))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_category[cat] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": support,
            }

        supports = [v["support"] for v in per_category.values()]
        active_cats = [v for v in per_category.values() if v["support"] > 0]
        macro_precision = float(np.mean([v["precision"] for v in active_cats])) if active_cats else 0.0
        macro_recall = float(np.mean([v["recall"] for v in active_cats])) if active_cats else 0.0
        macro_f1 = float(np.mean([v["f1"] for v in active_cats])) if active_cats else 0.0

        return {
            "n_samples": n_samples,
            "n_folds": n_folds,
            "per_category": per_category,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def predict(self, text: str) -> list[tuple[str, float]]:
        """Predict categories for new text."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before predicting")

        X = self.vectorizer.transform([text])
        categories = list(self.label_binarizer.classes_)
        results: list[tuple[str, float]] = []

        if SKLEARN_AVAILABLE:
            for idx, estimator in enumerate(self.classifier.estimators_):
                prob = float(estimator.predict_proba(X)[0, 1])
                results.append((categories[idx], prob))
        else:
            probas = self.classifier.predict_proba(X)[0]
            for idx, prob in enumerate(probas):
                results.append((categories[idx], float(prob)))

        results.sort(key=lambda x: x[1], reverse=True)
        filtered = [(cat, conf) for cat, conf in results if conf >= 0.25]
        if not filtered:
            filtered = [results[0]]

        return filtered

    def explain_prediction(self, text: str) -> dict[str, Any]:
        """Explain why categories were assigned to the given text."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before explaining")

        predictions = self.predict(text)
        X = self.vectorizer.transform([text])
        x_dense = X[0] if isinstance(X, np.ndarray) else X.toarray().flatten()
        categories = list(self.label_binarizer.classes_)
        explanations: dict[str, list[dict[str, float]]] = {}

        for cat, _conf in predictions:
            cat_idx = categories.index(cat)
            if SKLEARN_AVAILABLE:
                coefs = self.classifier.estimators_[cat_idx].coef_.flatten()
            else:
                coefs = self.classifier.weights_[:, cat_idx]

            contributions = x_dense * coefs
            top_indices = np.argsort(contributions)[::-1][:10]

            feature_explanations: list[dict[str, Any]] = []
            for i in top_indices:
                if contributions[i] <= 0:
                    break
                feature_explanations.append(
                    {
                        "feature": self._feature_names[i] if i < len(self._feature_names) else f"term_{i}",
                        "weight": float(contributions[i]),
                        "tfidf": float(x_dense[i]),
                        "coefficient": float(coefs[i]),
                    }
                )
            explanations[cat] = feature_explanations

        return {
            "predicted_categories": predictions,
            "explanations": explanations,
        }

    def evaluate(self, test_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate the model on a test set."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before evaluating")

        texts: list[str] = [item["text"] for item in test_data]
        y_true = self.label_binarizer.transform([item["categories"] for item in test_data])
        X = self.vectorizer.transform(texts)

        if SKLEARN_AVAILABLE:
            y_pred = self.classifier.predict(X)
        else:
            y_pred = self.classifier.predict(X)

        categories = list(self.label_binarizer.classes_)
        report_str = classification_report(y_true, y_pred, target_names=categories)

        per_category: dict[str, dict[str, Any]] = {}
        for idx, cat_name in enumerate(categories):
            yt_col = y_true[:, idx]
            yp_col = y_pred[:, idx]
            tp = int(np.sum((yt_col == 1) & (yp_col == 1)))
            fp = int(np.sum((yt_col == 0) & (yp_col == 1)))
            fn = int(np.sum((yt_col == 1) & (yp_col == 0)))
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            per_category[cat_name] = {
                "precision": p,
                "recall": r,
                "f1": f,
                "support": int(yt_col.sum()),
            }

        return {
            "classification_report": report_str,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
            "per_category": per_category,
            "n_test_samples": len(test_data),
        }

    def save(self, path: Path | str) -> None:
        """Save the trained classifier state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier,
                    "label_binarizer": self.label_binarizer,
                    "is_trained": self.is_trained,
                    "feature_names": self._feature_names,
                    "sklearn_backend": SKLEARN_AVAILABLE,
                },
                f,
            )

    def load(self, path: Path | str) -> None:
        """Load a trained classifier state from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.label_binarizer = data["label_binarizer"]
        self.is_trained = data["is_trained"]
        self._feature_names = data.get("feature_names", [])
