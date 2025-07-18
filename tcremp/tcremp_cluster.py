import sys

sys.path.append("../")
sys.path.append("../../mirpy")

import numpy as np
import pandas as pd
import time
import argparse
import logging
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from tcremp.utils import log_memory_usage
import faiss


def standardize_data(data: np.ndarray):
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
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    elapsed = time.time() - start
    logging.info(f"PCA completed: {n_components} components, time: {elapsed:.2f} sec.")
    return reduced


def apply_pca_incremental(data, n_components=50, batch_size=100000):
    start = time.time()
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(data)
    reduced = ipca.transform(data)
    elapsed = time.time() - start
    logging.info(f"Incremental PCA completed, time: {elapsed:.2f} sec.")
    return reduced


def get_k_neighbors_distance_matrix(data, n_neighbors=4):
    data = np.ascontiguousarray(data.astype('float32'))
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)

    squared_distances, _ = index.search(data, n_neighbors)
    distances = np.sqrt(squared_distances)
    return distances


def estimate_dbscan_eps(data, distances, poly_degree=10):
    total_start = time.time()

    total_num = len(data)
    number_of_points_for_knee = min(total_num, max(20000, int(total_num * 0.2)))
    chosen_elements = np.random.choice(distances, size=number_of_points_for_knee)
    distances_sorted = np.sort(chosen_elements)

    knee = KneeLocator(range(1, len(distances_sorted) + 1),
                       distances_sorted,
                       S=1.0,
                       curve="concave",
                       interp_method="polynomial",
                       polynomial_degree=poly_degree,
                       online=True,
                       direction="increasing", )

    eps = distances_sorted[knee.knee]
    logging.info(f"Estimated eps for DBSCAN: {eps:.4f}, total time: {(time.time() - total_start):.2f} sec.")
    return eps


def cluster_dbscan_with_filter(data, eps, min_samples, d1):
    try:
        data = data.to_numpy()
    except Exception:
        pass
    n_total = data.shape[0]
    start = time.time()
    mask = d1 <= eps
    n_filtered_out = np.sum(~mask)
    logging.info(
        f"Filtered out {n_filtered_out} points out of {n_total} ({n_filtered_out / n_total:.2%}) due to large d1 > eps")
    filtered_data = data[mask]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    filtered_labels = db.fit_predict(filtered_data)

    labels = np.full(data.shape[0], -1, dtype=int)
    labels[mask] = filtered_labels

    elapsed = time.time() - start
    n_clusters = len(set(filtered_labels)) - (1 if -1 in filtered_labels else 0)
    n_noise = list(labels).count(-1)
    logging.info(
        f"Filtered DBSCAN completed: clusters = {n_clusters}, noise points = {n_noise}, time: {elapsed:.2f} sec.")
    return labels


def prepare_data_for_clustering(df: pd.DataFrame, n_components):
    df = standardize_data(df.values)
    log_memory_usage('after standardization')

    df = apply_pca(df, n_components=n_components)
    log_memory_usage('after reduction')
    return df


def run_dbscan_clustering(df: pd.DataFrame, eps, closest_neigh_dist_array, min_samples: int = 5):
    labels = cluster_dbscan_with_filter(df, eps=eps, min_samples=min_samples, d1=closest_neigh_dist_array)
    log_memory_usage('after dbscan')
    return labels


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="Run clustering using PCA + DBSCAN")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--output", type=str, help="Path to output CSV file")
    parser.add_argument("--components", type=int, default=50, help="Number of PCA components (default: 50)")
    parser.add_argument("--min_samples", type=int, default=5, help="min_samples parameter for DBSCAN (default: 5)")
    parser.add_argument("--kth_neighbor", type=int, default=4,
                        help="k-th neighbor parameter for Knee estimation (default: 4)")
    args = parser.parse_args()

    logging.info("Loading data...")
    df = pd.read_parquet(args.input)

    logging.info('Preparing data for clustering...')
    df = prepare_data_for_clustering(df, n_components=args.components)

    logging.info('Evaluating k-neighbors distance matrix...')
    distances = get_k_neighbors_distance_matrix(df, n_neighbors=args.kth_neighbor)

    logging.info('Estimating epsilon for dbscan...')
    eps = estimate_dbscan_eps(df, distances=distances[:, args.kth_neighbor - 1])

    logging.info("Starting clustering...")
    labels = run_dbscan_clustering(df,
                                   eps=eps,
                                   closest_neigh_dist_array=distances[:, 1],
                                   min_samples=args.min_samples)

    df["cluster"] = labels
    df.to_csv(args.output, sep='\t')
    logging.info(f"Clustering results saved to {args.output}")


if __name__ == "__main__":
    main()
