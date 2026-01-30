"""Klasteryzacja embeddingów obrazów."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None


@dataclass
class ClusteringArtifacts:
    scaler: StandardScaler
    pca: PCA
    model: object


def _flatten_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim > 2:
        return embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


def select_optimal_pca_components(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int = 128,
    min_components: int = 16,
    sample_size: int = 5000,
    scale: bool = True,
) -> Tuple[int, plt.Figure]:
    """Dobiera liczbę komponentów PCA na podstawie wariancji."""
    features = _flatten_embeddings(embeddings)
    if scale:
        features = StandardScaler().fit_transform(features)
    if sample_size and len(features) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(features), size=sample_size, replace=False)
        features = features[indices]

    max_components = min(max_components, features.shape[1], features.shape[0])
    pca_full = PCA(n_components=max_components)
    pca_full.fit(features)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_n_components = int(np.argmax(cumulative_variance >= variance_threshold) + 1)
    optimal_n_components = max(min_components, optimal_n_components)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(1, max_components + 1), pca_full.explained_variance_ratio_ * 100)
    ax1.axvline(
        x=optimal_n_components,
        color="r",
        linestyle="--",
        label=f"Wybrane: {optimal_n_components}",
    )
    ax1.set_xlabel("Komponent PCA")
    ax1.set_ylabel("Wyjaśniona wariancja (%)")
    ax1.set_title("Wariancja per komponent")
    ax1.legend()

    ax2.plot(range(1, max_components + 1), cumulative_variance * 100, "b-")
    ax2.axhline(
        y=variance_threshold * 100,
        color="g",
        linestyle="--",
        label=f"Próg: {variance_threshold*100:.0f}%",
    )
    ax2.axvline(x=optimal_n_components, color="r", linestyle="--")
    ax2.scatter(
        [optimal_n_components],
        [cumulative_variance[optimal_n_components - 1] * 100],
        color="r",
        s=100,
        zorder=5,
    )
    ax2.set_xlabel("Liczba komponentów")
    ax2.set_ylabel("Skumulowana wariancja (%)")
    ax2.set_title(
        f"Optymalna liczba: {optimal_n_components} "
        f"({cumulative_variance[optimal_n_components - 1]*100:.1f}%)"
    )
    ax2.legend()

    plt.tight_layout()
    print(f"Wybrano {optimal_n_components} komponentów PCA")
    print(f"Wyjaśniona wariancja: {cumulative_variance[optimal_n_components - 1]*100:.2f}%")
    return optimal_n_components, fig


def build_embeddings(
    embeddings: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    max_components: int = 128,
    min_components: int = 16,
) -> Tuple[np.ndarray, StandardScaler, PCA]:
    features = _flatten_embeddings(embeddings)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    if n_components is None and variance_threshold is not None:
        n_components, _ = select_optimal_pca_components(
            scaled,
            variance_threshold=variance_threshold,
            max_components=max_components,
            min_components=min_components,
            scale=False,
        )
    if n_components is None:
        n_components = min(64, features.shape[1])
    pca = PCA(n_components=min(n_components, features.shape[1]))
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
    pca_components: Optional[int] = None,
    pca_variance_threshold: Optional[float] = None,
    pca_max_components: int = 128,
    pca_min_components: int = 16,
) -> Tuple[np.ndarray, ClusteringArtifacts]:
    reduced, scaler, pca = build_embeddings(
        embeddings,
        n_components=pca_components,
        variance_threshold=pca_variance_threshold,
        max_components=pca_max_components,
        min_components=pca_min_components,
    )
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


def compare_clusters_with_labels(
    labels: np.ndarray,
    styles: np.ndarray,
    artists: np.ndarray,
) -> Tuple[Dict[str, float], plt.Figure]:
    """Porównuje klastry z etykietami stylu i artysty."""
    ari_style = adjusted_rand_score(styles, labels)
    nmi_style = normalized_mutual_info_score(styles, labels)
    ari_artist = adjusted_rand_score(artists, labels)
    nmi_artist = normalized_mutual_info_score(artists, labels)

    metrics = {
        "ari_style": float(ari_style),
        "nmi_style": float(nmi_style),
        "ari_artist": float(ari_artist),
        "nmi_artist": float(nmi_artist),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        ["ARI style", "NMI style", "ARI artist", "NMI artist"],
        [ari_style, nmi_style, ari_artist, nmi_artist],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Porównanie klastrów z etykietami")
    ax.set_ylabel("Wynik")
    plt.tight_layout()
    return metrics, fig


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


def _find_elbow(k_values: np.ndarray, inertias: np.ndarray) -> int:
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    if len(second_diffs) == 0:
        return int(k_values[0])
    elbow_idx = int(np.argmax(second_diffs) + 2)
    return int(k_values[min(elbow_idx, len(k_values) - 1)])


def select_optimal_clusters(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    pca_components: Optional[int] = None,
    pca_variance_threshold: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[int, plt.Figure]:
    reduced, _, _ = build_embeddings(
        embeddings,
        n_components=pca_components,
        variance_threshold=pca_variance_threshold,
    )

    k_values = np.arange(k_min, k_max + 1)
    inertias = []
    silhouettes = []
    calinski_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(reduced)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(reduced, labels))
        calinski_scores.append(calinski_harabasz_score(reduced, labels))

    inertias_np = np.array(inertias)
    silhouettes_np = np.array(silhouettes)
    calinski_np = np.array(calinski_scores)

    k_elbow = _find_elbow(k_values, inertias_np)
    k_silhouette = int(k_values[int(np.argmax(silhouettes_np))])
    k_calinski = int(k_values[int(np.argmax(calinski_np))])

    votes = [k_elbow, k_silhouette, k_calinski]
    optimal_k = max(set(votes), key=votes.count)
    if votes.count(optimal_k) == 1:
        optimal_k = k_silhouette

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(k_values, inertias_np, marker="o")
    axes[0].axvline(optimal_k, color="r", linestyle="--")
    axes[0].set_title("Elbow (inercja)")
    axes[0].set_xlabel("Liczba klastrów")

    axes[1].plot(k_values, silhouettes_np, marker="o")
    axes[1].axvline(optimal_k, color="r", linestyle="--")
    axes[1].set_title("Silhouette")
    axes[1].set_xlabel("Liczba klastrów")

    axes[2].plot(k_values, calinski_np, marker="o")
    axes[2].axvline(optimal_k, color="r", linestyle="--")
    axes[2].set_title("Calinski-Harabasz")
    axes[2].set_xlabel("Liczba klastrów")

    plt.tight_layout()
    print(f"Elbow: {k_elbow}, Silhouette: {k_silhouette}, Calinski: {k_calinski}")
    print(f"Wybrano: {optimal_k} klastrów")
    return optimal_k, fig


def generate_cluster_visualization(
    embeddings: np.ndarray,
    n_clusters: int,
    method: str = "both",
    random_state: int = 42,
    max_samples: int = 10000,
) -> Dict[str, plt.Figure]:
    reduced, _, _ = build_embeddings(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced)

    figures: Dict[str, plt.Figure] = {}

    if method in {"umap", "both"}:
        umap_embedding = reduce_umap(reduced, n_components=2)
        fig_umap = plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=cluster_labels,
            cmap="viridis",
            s=5,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Klaster")
        plt.title(f"UMAP - {n_clusters} klastrów")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        figures["umap"] = fig_umap

    if method in {"tsne", "both"}:
        if len(reduced) > max_samples:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(reduced), size=max_samples, replace=False)
            reduced_sample = reduced[indices]
            labels_sample = cluster_labels[indices]
        else:
            reduced_sample = reduced
            labels_sample = cluster_labels

        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=30,
            learning_rate="auto",
            init="pca",
        )
        tsne_embedding = tsne.fit_transform(reduced_sample)

        fig_tsne = plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            tsne_embedding[:, 0],
            tsne_embedding[:, 1],
            c=labels_sample,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Klaster")
        plt.title(f"t-SNE - {n_clusters} klastrów")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        figures["tsne"] = fig_tsne

    return figures
