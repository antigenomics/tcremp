import argparse

from tcremp.constants import DEFAULT_PAIRED_CHAIN_COMPONENTS, SUPPORTED_CHAINS, SUPPORTED_SINGLE_CHAINS


def add_common_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the main input file.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output folder or output file, depending on the command.")
    parser.add_argument("-e", "--prefix", type=str,
                        help="Output prefix. Defaults to the input filename stem.")
    return parser


def add_run_metadata_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-x", "--index-col", type=str,
                        help="Column containing stable clonotype IDs to transfer to outputs.")
    parser.add_argument("-l", "--labels-col", type=str,
                        help="Column containing labels for t-SNE coloring.")
    parser.add_argument("--enrich-by", type=str,
                        help="Column used for cluster enrichment analysis.")
    return parser


def add_common_embedding_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    paired_modes = ", ".join(f'"{x}"' for x in DEFAULT_PAIRED_CHAIN_COMPONENTS)
    parser.add_argument("-c", "--chain", type=str, required=True,
                        choices=list(SUPPORTED_CHAINS),
                        help="Single-chain input supports "
                             + ", ".join(f'\"{x}\"' for x in SUPPORTED_SINGLE_CHAINS)
                             + f", while paired-chain input uses {paired_modes}. "
                               "Built-in prototypes are resolved from per-chain resource files when available.")
    parser.add_argument("-p", "--prototypes-path", type=str,
                        help="Path to a user-specified prototypes file.")
    parser.add_argument("-n", "--n-prototypes", type=int,
                        help="Number of prototypes to use for embedding.")
    parser.add_argument("-sample_random_p", "--sample-random-prototypes", action="store_true",
                        help="Sample prototypes randomly.")
    parser.add_argument("-nc", "--n-clonotypes", type=int,
                        help="Number of clonotypes to process.")
    parser.add_argument("-sample_random_c", "--sample-random-clonotypes", action="store_true",
                        help="Sample clonotypes randomly.")
    parser.add_argument("-s", "--species", type=str, default="HomoSapiens",
                        choices=["HomoSapiens", "MusMusculus", "MacacaMulatta"],
                        help="Species for V/J gene alignment.")
    parser.add_argument("-u", "--unique-clonotypes", action="store_true",
                        help="Run only on unique clonotypes/clones from the input table.")
    parser.add_argument("-r", "--random-seed", type=int, default=42,
                        help="Random seed for sampling and stochastic procedures.")
    parser.add_argument("-np", "--nproc", type=int, default=None,
                        help="Number of worker threads/processes to use. Defaults to auto.")
    parser.add_argument("-llen", "--lower-len-cdr3", type=int, default=5,
                        help="Filter out CDR3 with length below this threshold.")
    parser.add_argument("-hlen", "--higher-len-cdr3", type=int, default=30,
                        help="Filter out CDR3 with length greater than or equal to this threshold.")
    parser.add_argument("-m", "--metrics", type=str, default="dissimilarity",
                        choices=["similarity", "dissimilarity"],
                        help="Whether to calculate similarity or dissimilarity scores.")
    parser.add_argument("-d", "--save-dists", action="store_true", default=True,
                        help="Save the file with evaluated TCRemP distances.")
    parser.add_argument("--no-save-dists", dest="save_dists", action="store_false",
                        help="Do not save the file with evaluated TCRemP distances.")
    return parser


def add_common_clustering_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--skip-clustering", action="store_true",
                        help="Skip clustering.")
    parser.add_argument("-npc", "--cluster-pc-components", type=int, default=50,
                        help="Number of PCA components for clustering.")
    parser.add_argument("-ms", "--cluster-min-samples", type=int, default=3,
                        help="min_samples parameter for DBSCAN.")
    parser.add_argument("-kn", "--k-neighbors", type=int, default=4,
                        help="k-th neighbor parameter for knee-based eps estimation.")
    return parser


def add_common_tsne_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--tsne", action="store_true",
                        help="Run PCA+t-SNE visualization pipeline and save coordinates.")
    parser.add_argument("--tsne-init", type=str, default="pca", choices=["pca", "random"],
                        help="Initialization for t-SNE.")
    parser.add_argument("--tsne-perplexity", type=float, default=15,
                        help="Perplexity for t-SNE.")
    return parser


def add_common_enrichment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--enrichment-threshold", type=float, default=0.7,
                        help="Advisory within-cluster label fraction threshold reported in output.")
    parser.add_argument("--enrichment-fdr-threshold", type=float, default=0.05,
                        help="FDR threshold used as the main enrichment cutoff.")
    return parser


def add_cluster_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--components", type=int, default=50,
                        help="Number of PCA components.")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="min_samples parameter for DBSCAN.")
    parser.add_argument("--kth_neighbor", type=int, default=4,
                        help="k-th neighbor parameter for knee estimation.")
    return parser


def add_enrich_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-l", "--label-col", required=True,
                        help="Column used for enrichment analysis.")
    parser.add_argument("--cluster-col", default="cluster_id",
                        help="Cluster column name.")
    return parser


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General TCRemP embedding pipeline")
    add_common_io_args(parser)
    add_run_metadata_args(parser)
    add_common_embedding_args(parser)
    add_common_clustering_args(parser)
    add_common_tsne_args(parser)
    add_common_enrichment_args(parser)
    return parser


def build_enrich_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute label enrichment for existing TCRemP clusters")
    add_common_io_args(parser)
    add_enrich_cli_args(parser)
    add_common_enrichment_args(parser)
    return parser


def build_cluster_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clustering using PCA + DBSCAN")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input data file (CSV/TSV).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the clustering results.")
    add_cluster_cli_args(parser)
    return parser


def get_arguments(args=None):
    return build_run_parser().parse_args(args)


def get_arguments_enrich(args=None):
    return build_enrich_parser().parse_args(args)


def get_arguments_cluster(args=None):
    return build_cluster_parser().parse_args(args)

