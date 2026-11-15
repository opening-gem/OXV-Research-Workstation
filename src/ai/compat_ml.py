"""
Machine learning and text processing utilities for Flame in Freefall.
Provides pure Python and NumPy implementations of TF-IDF vectorization,
cosine similarity, and classification as fallbacks when scikit-learn is not installed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np

# Standard English stop words
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "almost", "alone",
    "along", "already", "also", "although", "always", "am", "among", "amongst",
    "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway",
    "anywhere", "are", "area", "areas", "around", "as", "ask", "asked", "asking",
    "asks", "at", "away", "b", "back", "backed", "backing", "backs", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before",
    "beforehand", "began", "behind", "being", "beings", "below", "beside",
    "besides", "best", "better", "between", "beyond", "big", "both", "but", "by",
    "c", "came", "can", "cannot", "case", "cases", "certain", "certainly",
    "clear", "clearly", "come", "could", "d", "did", "differ", "different",
    "differently", "do", "does", "done", "down", "downed", "downing", "downs",
    "during", "e", "each", "early", "either", "end", "ended", "ending", "ends",
    "enough", "even", "evenly", "ever", "every", "everybody", "everyone",
    "everything", "everywhere", "f", "face", "faces", "fact", "facts", "far",
    "felt", "few", "fewer", "find", "finds", "first", "for", "four", "from",
    "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave",
    "general", "generally", "get", "gets", "give", "given", "gives", "go",
    "going", "good", "goods", "got", "great", "greater", "greatest", "group",
    "grouped", "grouping", "groups", "h", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "however",
    "i", "if", "important", "in", "interest", "interested", "interesting",
    "interests", "into", "is", "it", "its", "itself", "j", "just", "k", "keep",
    "keeps", "kind", "knew", "know", "known", "knows", "l", "large", "largely",
    "last", "later", "latest", "least", "less", "let", "lets", "like", "likely",
    "long", "longer", "longest", "m", "made", "make", "making", "man", "many",
    "may", "me", "member", "members", "men", "might", "more", "most", "mostly",
    "mr", "mrs", "much", "must", "my", "myself", "n", "necessary", "need",
    "needed", "needing", "needs", "never", "new", "newer", "newest", "next",
    "no", "nobody", "non", "noone", "not", "nothing", "now", "nowhere", "number",
    "numbers", "o", "of", "off", "often", "old", "older", "oldest", "on", "once",
    "one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered",
    "ordering", "orders", "other", "others", "our", "out", "over", "p", "part",
    "parted", "parting", "parts", "per", "perhaps", "place", "places", "point",
    "pointed", "pointing", "points", "possible", "present", "presented",
    "presenting", "presents", "problem", "problems", "put", "puts", "q", "quite",
    "r", "rather", "really", "right", "room", "rooms", "s", "said",
    "saw", "say", "says", "second", "seconds", "see", "seem", "seemed", "seeming",
    "seems", "sees", "several", "shall", "she", "should", "show", "showed",
    "showing", "shows", "side", "sides", "since", "small", "smaller", "smallest",
    "so", "some", "somebody", "someone", "something", "somewhere", "state",
    "states", "still", "such", "sure", "t", "take", "taken", "than", "that",
    "the", "their", "them", "then", "there", "therefore", "these", "they",
    "thing", "things", "think", "thinks", "this", "those", "though", "thought",
    "thoughts", "three", "through", "throughout", "thru", "thus", "to", "today",
    "together", "too", "took", "toward", "turn", "turned", "turning", "turns",
    "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses",
    "v", "very", "w", "want", "wanted", "wanting", "wants", "was", "way", "ways",
    "we", "well", "wells", "went", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whole", "whose", "why", "will", "with", "within",
    "without", "work", "worked", "working", "works", "would", "x", "y", "year",
    "years", "yet", "you", "young", "younger", "youngest", "your", "yours", "z"
}


def _tokenize(text: str, ngram_range: Tuple[int, int] = (1, 1), stop_words: Any = None) -> List[str]:
    """Extract lowercased word tokens and n-grams matching word characters."""
    words = re.findall(r"(?u)\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    if stop_words == "english":
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    elif stop_words and hasattr(stop_words, "__contains__"):
        words = [w for w in words if w not in stop_words]

    tokens = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if n == 1:
            tokens.extend(words)
        else:
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i : i + n]))
    return tokens


class TfidfVectorizer:
    """
    Self-built TF-IDF Vectorizer with sublinear term-frequency scaling and L2 normalization.
    """
    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: Tuple[int, int] = (1, 2),
        stop_words: str | Sequence[str] | None = "english",
        sublinear_tf: bool = True,
        norm: str = "l2",
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_: np.ndarray | None = None

    def fit(self, raw_documents: Sequence[str]) -> "TfidfVectorizer":
        doc_count = len(raw_documents)
        df_counter: Counter[str] = Counter()

        for doc in raw_documents:
            tokens = set(_tokenize(doc, self.ngram_range, self.stop_words))
            for t in tokens:
                df_counter[t] += 1

        # Select top features by frequency
        if self.max_features is not None:
            most_common = df_counter.most_common(self.max_features)
        else:
            most_common = sorted(df_counter.items(), key=lambda x: (-x[1], x[0]))

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(most_common)}
        self.feature_names_ = [term for term, _ in most_common]

        # Calculate smooth IDF: log((1 + n) / (1 + df)) + 1
        idfs = []
        for term, _ in most_common:
            df = df_counter[term]
            idf = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0
            idfs.append(idf)
        self.idf_ = np.array(idfs, dtype=np.float64)
        return self

    def transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        if self.idf_ is None or not self.vocabulary_:
            raise ValueError("TfidfVectorizer is not fitted.")

        num_docs = len(raw_documents)
        num_features = len(self.feature_names_)
        matrix = np.zeros((num_docs, num_features), dtype=np.float64)

        for i, doc in enumerate(raw_documents):
            tokens = _tokenize(doc, self.ngram_range, self.stop_words)
            tf_counter = Counter(tokens)
            for term, count in tf_counter.items():
                col = self.vocabulary_.get(term)
                if col is not None:
                    # sublinear tf
                    tf = 1.0 + math.log(count) if self.sublinear_tf and count > 0 else float(count)
                    matrix[i, col] = tf * self.idf_[col]

        # L2 normalize
        if self.norm == "l2":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms

        return matrix

    def fit_transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_)


def cosine_similarity(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute cosine similarity between two matrices of vectors."""
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    if len(Y.shape) == 1:
        Y = Y.reshape(1, -1)

    norm_x = np.linalg.norm(X, axis=1, keepdims=True)
    norm_y = np.linalg.norm(Y, axis=1, keepdims=True)
    norm_x[norm_x == 0.0] = 1.0
    norm_y[norm_y == 0.0] = 1.0

    X_norm = X / norm_x
    Y_norm = Y / norm_y
    return np.dot(X_norm, Y_norm.T)


def normalize(X: np.ndarray, norm: str = "l2", axis: int = 1) -> np.ndarray:
    """Normalize vectors along given axis."""
    if norm == "l2":
        norms = np.linalg.norm(X, axis=axis, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms
    return X


class MultiLabelBinarizer:
    """Convert a sequence of label collections to binary matrix format."""
    def __init__(self):
        self.classes_: list[str] = []
        self._class_to_idx: dict[str, int] = {}

    def fit(self, y: Sequence[Sequence[str]]) -> "MultiLabelBinarizer":
        classes = sorted({c for labels in y for c in labels})
        self.classes_ = classes
        self._class_to_idx = {c: i for i, c in enumerate(classes)}
        return self

    def transform(self, y: Sequence[Sequence[str]]) -> np.ndarray:
        matrix = np.zeros((len(y), len(self.classes_)), dtype=int)
        for i, labels in enumerate(y):
            for label in labels:
                idx = self._class_to_idx.get(label)
                if idx is not None:
                    matrix[i, idx] = 1
        return matrix

    def fit_transform(self, y: Sequence[Sequence[str]]) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, yt: np.ndarray) -> list[list[str]]:
        result = []
        for row in yt:
            labels = [self.classes_[i] for i, val in enumerate(row) if val > 0]
            result.append(labels)
        return result


class OneVsRestLogisticRegression:
    """
    Multi-label Logistic Regression classifier with Ridge (L2) regularization.
    Trains one binary logistic regression model per class.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 200, learning_rate: float = 0.1):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights_: np.ndarray | None = None  # (num_features, num_classes)
        self.intercept_: np.ndarray | None = None  # (num_classes,)
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OneVsRestLogisticRegression":
        num_samples, num_features = X.shape
        num_classes = Y.shape[1]
        self.weights_ = np.zeros((num_features, num_classes), dtype=np.float64)
        self.intercept_ = np.zeros(num_classes, dtype=np.float64)

        # Train each binary classifier using regularized gradient descent
        for c in range(num_classes):
            y_c = Y[:, c].astype(np.float64)
            w = np.zeros(num_features, dtype=np.float64)
            b = 0.0
            
            # If all 0s or all 1s, initialize bias
            pos_ratio = np.mean(y_c)
            if pos_ratio > 0 and pos_ratio < 1:
                b = math.log(pos_ratio / (1.0 - pos_ratio))

            l2_reg = 1.0 / (self.C * num_samples)

            for _ in range(self.max_iter):
                linear = np.dot(X, w) + b
                # Sigmoid with clipping for numerical stability
                linear = np.clip(linear, -15.0, 15.0)
                p = 1.0 / (1.0 + np.exp(-linear))

                error = p - y_c
                grad_w = (np.dot(X.T, error) / num_samples) + l2_reg * w
                grad_b = np.mean(error)

                w -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b

            self.weights_[:, c] = w
            self.intercept_[c] = b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.intercept_ is None:
            raise ValueError("Classifier is not fitted.")
        linear = np.dot(X, self.weights_) + self.intercept_
        linear = np.clip(linear, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-linear))

    def predict(self, X: np.ndarray, threshold: float = 0.4) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients transposed to match sklearn shape: (num_classes, num_features)."""
        return self.weights_.T if self.weights_ is not None else np.array([])


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute precision."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    precisions = []
    for c in range(y_true.shape[1]):
        tp = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1))
        fp = np.sum((y_true[:, c] == 0) & (y_pred[:, c] == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(p)
    return float(np.mean(precisions)) if average == "macro" else float(precisions[0])


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute recall."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    recalls = []
    for c in range(y_true.shape[1]):
        tp = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1))
        fn = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 0))
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(r)
    return float(np.mean(recalls)) if average == "macro" else float(recalls[0])


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute F1 score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    f1s = []
    for c in range(y_true.shape[1]):
        tp = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1))
        fp = np.sum((y_true[:, c] == 0) & (y_pred[:, c] == 1))
        fn = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if average == "macro" else float(f1s[0])


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None) -> str:
    """Generate human-readable classification report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lines = [f"{'Class':<28} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}"]
    lines.append("-" * 68)
    
    macro_p, macro_r, macro_f = [], [], []
    total_support = 0

    for c in range(y_true.shape[1]):
        name = target_names[c] if target_names and c < len(target_names) else f"Class_{c}"
        tp = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1))
        fp = np.sum((y_true[:, c] == 0) & (y_pred[:, c] == 1))
        fn = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 0))
        support = int(np.sum(y_true[:, c] == 1))
        total_support += support

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        macro_p.append(p)
        macro_r.append(r)
        macro_f.append(f)

        lines.append(f"{name:<28} {p:<10.3f} {r:<10.3f} {f:<10.3f} {support:<8}")

    lines.append("-" * 68)
    lines.append(f"{'Macro Average':<28} {np.mean(macro_p):<10.3f} {np.mean(macro_r):<10.3f} {np.mean(macro_f):<10.3f} {total_support:<8}")
    return "\n".join(lines)
