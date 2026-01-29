"""Klasteryzacja embeddingów obrazów."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None


@dataclass
class ClusteringArtifacts:
    scaler: StandardScaler
    pca: PCA
    model: object


def build_embeddings(embeddings: np.ndarray) -> Tuple[np.ndarray, StandardScaler, PCA]:
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=min(64, embeddings.shape[1]))
    reduced = pca.fit_transform(scaled)
    return reduced, scaler, pca


def reduce_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Redukuje embeddingi do 2D/3D przy pomocy UMAP."""
    if umap is None:
        raise ImportError("UMAP nie jest dostępny. Zainstaluj umap-learn.")
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 8,
) -> Tuple[np.ndarray, ClusteringArtifacts]:
    reduced, scaler, pca = build_embeddings(embeddings)
    if method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(reduced)
    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(reduced)
    elif method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(reduced)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(reduced)
    return labels, ClusteringArtifacts(scaler=scaler, pca=pca, model=model)


def evaluate_clusters(labels: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Podstawowa metryka porównania klastrów z etykietami."""
    score = normalized_mutual_info_score(ground_truth, labels)
    return {"nmi": float(score)}


def summarize_labels(labels: np.ndarray, metadata: np.ndarray) -> Dict[int, Dict[str, int]]:
    """Zwraca top etykiety dla każdego klastra (np. style/autor)."""
    summary: Dict[int, Dict[str, int]] = {}
    for label in np.unique(labels):
        items = metadata[labels == label]
        counts: Dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        summary[int(label)] = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
    return summary
