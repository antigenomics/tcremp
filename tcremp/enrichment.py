import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _binom_pvalue_greater(k, n, p):
    if hasattr(stats, "binomtest"):
        return stats.binomtest(k, n=n, p=p, alternative="greater").pvalue
    return stats.binom_test(k, n=n, p=p, alternative="greater")


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted[i] = min(prev, ranked[i] * n / rank)
        prev = adjusted[i]

    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def compute_cluster_enrichment(
    cluster_df: pd.DataFrame,
    label_col: str,
    cluster_col: str = "cluster_id",
    threshold: float = 0.7,
    fdr_threshold: float = 0.05,
):
    if cluster_col not in cluster_df.columns:
        raise ValueError(f"Missing cluster column: {cluster_col}")
    if label_col not in cluster_df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    df = cluster_df[[cluster_col, label_col]].copy().dropna(subset=[label_col])
    if df.empty:
        raise ValueError(f"No non-null values found in label column: {label_col}")

    df["total_cluster"] = df.groupby(cluster_col)[cluster_col].transform("count")
    df["total_group"] = df.groupby(label_col)[label_col].transform("count")
    df["count_matched"] = df.groupby([label_col, cluster_col])[label_col].transform("count")
    df["fraction_matched"] = df["count_matched"] / df["total_cluster"]
    df["fraction_matched_exp"] = df["total_group"] / len(df.index)
    df["enrichment_pvalue"] = df.apply(
        lambda row: _binom_pvalue_greater(
            int(row["count_matched"]),
            int(row["total_cluster"]),
            float(row["fraction_matched_exp"]),
        ),
        axis=1,
    )

    summary = (
        df[
            [
                label_col,
                cluster_col,
                "total_cluster",
                "total_group",
                "count_matched",
                "fraction_matched",
                "fraction_matched_exp",
                "enrichment_pvalue",
            ]
        ]
        .drop_duplicates()
        .sort_values(["fraction_matched", "enrichment_pvalue"], ascending=[False, True])
    )
    summary["is_cluster"] = ((summary["total_cluster"] > 1) & (summary[cluster_col] != -1)).astype(int)
    summary["enrichment_fdr"] = benjamini_hochberg(summary["enrichment_pvalue"].to_numpy())
    summary["enriched_cluster"] = (
        (summary["enrichment_fdr"] <= fdr_threshold) & (summary["is_cluster"] == 1)
    ).astype(int)
    summary["passes_fraction_threshold"] = (
        (summary["fraction_matched"] >= threshold) & (summary["is_cluster"] == 1)
    ).astype(int)

    best_per_cluster = (
        summary.drop_duplicates(subset=[cluster_col], keep="first")
        .rename(columns={label_col: "label_cluster"})
        .sort_values(cluster_col)
        .reset_index(drop=True)
    )
    return best_per_cluster


def annotate_clusters_with_enrichment(
    cluster_df: pd.DataFrame,
    enrichment_summary: pd.DataFrame,
    cluster_col: str = "cluster_id",
):
    return cluster_df.merge(enrichment_summary, on=cluster_col, how="left")


def save_enrichment_outputs(
    annotated_clusters: pd.DataFrame,
    enrichment_summary: pd.DataFrame,
    output_prefix: str,
    output_dir,
):
    output_dir = Path(output_dir)
    summary_path = output_dir / f"{output_prefix}_tcremp_enrichment_summary.tsv"
    clusters_path = output_dir / f"{output_prefix}_tcremp_clusters_enriched.tsv"
    enrichment_summary.to_csv(summary_path, sep="\t", index=False)
    annotated_clusters.to_csv(clusters_path, sep="\t", index=False)
    logging.info("Saved enrichment summary to %s", summary_path)
    logging.info("Saved enriched cluster annotations to %s", clusters_path)
    return summary_path, clusters_path
