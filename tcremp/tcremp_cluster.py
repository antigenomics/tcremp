import numpy as np
import pandas as pd
import time
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def standardize_data(data):
    start = time.time()
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data)
    elapsed = time.time() - start
    logging.info(f"Standardization completed, time: {elapsed:.2f} sec.")
    return standardized


def apply_pca(data, n_components=50):
    start = time.time()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    elapsed = time.time() - start
    logging.info(f"PCA completed: {n_components} components, time: {elapsed:.2f} sec.")
    return reduced


def estimate_dbscan_eps(data, n_neighbors=4, quantile=0.05, poly_degree=10):
    start = time.time()
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances[:, n_neighbors - 1])

    knee = KneeLocator(range(1, len(distances) + 1),  # x values
                       distances,  # y values
                       S=1.0,  # parameter suggested from paper
                       curve="concave",
                       interp_method="polynomial",
                       polynomial_degree=poly_degree,
                       online=True,
                       direction="increasing", )

    eps = distances[knee.knee] if knee.knee else distances[int(len(distances) * quantile)]
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


def run_dbscan_clustering_from_dataframe(df: pd.DataFrame, n_components: int = 50, min_samples: int = 5):
    # Standardize data
    standardized = standardize_data(df.values)

    # Apply PCA and reduce dimensionality
    reduced = apply_pca(standardized, n_components=n_components)

    # Estimate optimal eps using k-nearest neighbors and KneeLocator
    eps = estimate_dbscan_eps(reduced)

    # Run DBSCAN clustering
    labels = cluster_dbscan(reduced, eps=eps, min_samples=min_samples)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Run clustering using PCA + DBSCAN")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("output_file", type=str, help="Path to output CSV file")
    parser.add_argument("--components", type=int, default=50, help="Number of PCA components (default: 50)")
    parser.add_argument("--min_samples", type=int, default=5, help="min_samples parameter for DBSCAN (default: 5)")
    args = parser.parse_args()

    logging.info("Loading data...")
    df = pd.read_csv(args.input_file, sep='\t')

    logging.info("Starting clustering...")
    labels = run_dbscan_clustering_from_dataframe(df, n_components=args.components, min_samples=args.min_samples)

    df["cluster"] = labels
    df.to_csv(args.output_file, sep='\t')
    logging.info(f"Clustering results saved to {args.output_file}")


if __name__ == "__main__":
    main()
