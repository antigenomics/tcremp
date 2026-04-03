import numpy as np
import pandas as pd
import time
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

from tcremp.arguments import get_arguments_cluster


def standardize_data(data):
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32, copy=False)

    start = time.time()
    means = np.mean(data, axis=0, dtype=np.float32)
    stds = np.std(data, axis=0, dtype=np.float32)
    stds[stds == 0] = 1.0
    data -= means
    data /= stds
    elapsed = time.time() - start
    logging.info(f"Standardization (in-place, overflow-safe) completed in {elapsed:.2f} sec.")
    return data


def apply_pca(data, n_components=50):
    start = time.time()
    n_components = min(n_components, data.shape[0], data.shape[1])
    if n_components < 1:
        raise ValueError("PCA requires at least one sample and one feature.")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    elapsed = time.time() - start
    logging.info(f"PCA completed: {n_components} components, time: {elapsed:.2f} sec.")
    return reduced


def prepare_data_for_clustering(df: pd.DataFrame, n_components: int):
    standardized = standardize_data(df.values)
    return apply_pca(standardized, n_components=n_components)


def estimate_dbscan_eps(data, distances=None, n_neighbors=4, quantile=0.05, poly_degree=10):
    start = time.time()
    if distances is None:
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(data)
        distances, _ = nbrs.kneighbors(data)
        kth_distances = distances[:, n_neighbors - 1]
    else:
        kth_distances = np.asarray(distances)

    kth_distances = np.sort(kth_distances)

    knee = KneeLocator(
        range(1, len(kth_distances) + 1),
        kth_distances,
        S=1.0,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_degree,
        online=True,
        direction="increasing",
    )

    eps = kth_distances[knee.knee] if knee.knee is not None else kth_distances[int(len(kth_distances) * quantile)]
    elapsed = time.time() - start
    logging.info(f"Estimated eps for DBSCAN: {eps:.4f}, time: {elapsed:.2f} sec.")
    return eps


def cluster_dbscan(data, eps=None, min_samples=5):
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    elapsed = time.time() - start
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logging.info(
        f"DBSCAN completed: clusters = {n_clusters}, noise points = {n_noise}, time: {elapsed:.2f} sec."
    )
    return labels


def run_dbscan_clustering(df: pd.DataFrame, n_components: int = 50, min_samples: int = 5, n_neighbors: int = 4):
    reduced = prepare_data_for_clustering(df, n_components=n_components)
    eps = estimate_dbscan_eps(reduced, n_neighbors=n_neighbors)
    return cluster_dbscan(reduced, eps=eps, min_samples=min_samples)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    args = get_arguments_cluster()

    logging.info("Loading data...")
    df = pd.read_csv(args.input, sep='\t')

    logging.info("Starting clustering...")
    labels = run_dbscan_clustering(df,
                                   n_components=args.components,
                                   min_samples=args.min_samples,
                                   n_neighbors=args.kth_neighbor)

    df["cluster"] = labels
    df.to_csv(args.output, sep='\t', index=False)
    logging.info(f"Clustering results saved to {args.output}")


if __name__ == "__main__":
    main()
