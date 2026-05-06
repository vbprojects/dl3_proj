"""Modular KNN and Linear Probe classification heads for evaluation."""

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def evaluate_knn(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    n_neighbors: int = 5,
) -> float:
    """Fit a k-NN classifier on train embeddings and score on val embeddings.

    Args:
        train_embeddings: [N_train, D] float32 tensor.
        train_labels: [N_train] long tensor.
        val_embeddings: [N_val, D] float32 tensor.
        val_labels: [N_val] long tensor.
        n_neighbors: Number of neighbors for k-NN.

    Returns:
        Accuracy (float between 0 and 1).
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings.numpy(), train_labels.numpy())
    return float(knn.score(val_embeddings.numpy(), val_labels.numpy()))


def evaluate_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    max_iter: int = 1000,
    *,
    clf: Optional[LogisticRegression] = None,
) -> Tuple[float, LogisticRegression]:
    """Fit a linear probe (LogisticRegression) and return accuracy + fitted classifier.

    Args:
        train_embeddings: [N_train, D] float32 tensor.
        train_labels: [N_train] long tensor.
        val_embeddings: [N_val, D] float32 tensor.
        val_labels: [N_val] long tensor.
        max_iter: Maximum iterations for sklearn LogisticRegression solver.
        clf: Optional pre-allocated LogisticRegression instance to reuse memory.

    Returns:
        (accuracy, fitted_classifier)
    """
    if clf is None:
        clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    clf.fit(train_embeddings.numpy(), train_labels.numpy())
    acc = float(clf.score(val_embeddings.numpy(), val_labels.numpy()))
    return acc, clf


class ClassificationHeadEvaluator:
    """Caches train embeddings extracted once and re-evaluates each epoch."""

    def __init__(self, n_neighbors: int = 5, linear_probe_max_iter: int = 1000):
        self.n_neighbors = n_neighbors
        self.linear_probe_max_iter = linear_probe_max_iter
        self._fitted_knn: Optional[KNeighborsClassifier] = None
        self._train_embeddings: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None

    def fit_train(self, train_embeddings: torch.Tensor, train_labels: torch.Tensor) -> None:
        """Cache train embeddings as numpy arrays and fit k-NN once."""
        self._train_embeddings = train_embeddings.float().cpu().numpy()
        self._train_labels = train_labels.cpu().numpy()
        self._fitted_knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self._fitted_knn.fit(self._train_embeddings, self._train_labels)

    def knn_accuracy(self, val_embeddings: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Return k-NN accuracy using cached train embeddings."""
        if self._fitted_knn is None:
            raise RuntimeError("fit_train() must be called before knn_accuracy().")
        return float(
            self._fitted_knn.score(
                val_embeddings.float().cpu().numpy(),
                val_labels.cpu().numpy(),
            )
        )

    def linear_probe_accuracy(
        self,
        val_embeddings: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> float:
        """Fit a new linear probe each call (lightweight) and return accuracy."""
        if self._train_embeddings is None or self._train_labels is None:
            raise RuntimeError("fit_train() must be called before linear_probe_accuracy().")
        acc, _ = evaluate_linear_probe(
            torch.from_numpy(self._train_embeddings),
            torch.from_numpy(self._train_labels),
            val_embeddings,
            val_labels,
            max_iter=self.linear_probe_max_iter,
        )
        return acc