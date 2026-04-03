import sys

sys.path.append("../")

import os
import pandas as pd
import logging
import time

from mir.common.segments import SegmentLibrary
from mir.embedding.prototype_embedding import PrototypeEmbedding, Metrics
from mir.distances.aligner import ClonotypeAligner

from tcremp.arguments import get_arguments
from tcremp.enrichment import (
    annotate_clusters_with_enrichment,
    compute_cluster_enrichment,
    save_enrichment_outputs,
)
from tcremp.tcremp_cluster import run_dbscan_clustering
from tcremp.utils import (
    configure_logging,
    generate_output_prefix,
    get_metadata_columns,
    get_representations_df,
    load_analysis_repertoire,
    load_prototype_repertoire,
    log_memory_usage,
    prepare_output_path,
    resolve_input_file,
    resolve_prototype_file,
    subsample_repertoire,
    make_custom_palette,
    pca_proc,
    tsne_plot,
    tsne_proc,
)

def _safe_embedding_threads(nproc, n_clonotypes, min_batch_size=32):
    if n_clonotypes <= 0:
        return 0
    if nproc is None:
        nproc = 1
    by_batch = max(1, n_clonotypes // max(1, int(min_batch_size)))
    return max(1, min(int(nproc), int(n_clonotypes), by_batch))


def run_tcremp_embedding(analysis_rep, proto_rep, segment_library, chain, metrics, nproc):
    aligner = ClonotypeAligner.from_library(lib=segment_library)
    logging.info("Started embeddings calculation")
    embedder = PrototypeEmbedding(proto_rep, aligner=aligner, metrics=Metrics(metrics))
    n_clonotypes = len(analysis_rep.clonotypes)
    if n_clonotypes == 0:
        raise ValueError(
            "Cannot compute embeddings for an empty repertoire. "
            "All clonotypes were filtered out before embedding."
        )

    threads = _safe_embedding_threads(nproc, n_clonotypes)
    logging.info(
        "Embedding repertoire with %d clonotypes using %d worker(s) (requested nproc=%s)",
        n_clonotypes,
        threads,
        nproc,
    )
    t0 = time.time()
    emb = embedder.embed_repertoire(analysis_rep, threads=threads, flatten_scores=True)
    logging.info("Embeddings done in %.2fs", time.time() - t0)
    log_memory_usage("after embeddings done")

    columns = []
    for i in range(proto_rep.total):
        if "TRA" in chain:
            columns += [f"{i}_a_v", f"{i}_a_j", f"{i}_a_cdr3"]
        if "TRB" in chain:
            columns += [f"{i}_b_v", f"{i}_b_j", f"{i}_b_cdr3"]

    df = pd.DataFrame(emb, columns=columns)
    log_memory_usage("after embeddings dataframe creation")
    return df


def main():
    args = get_arguments()

    input_path = resolve_input_file(args.input)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    output_prefix = generate_output_prefix(args.input, args.prefix)
    configure_logging(input_path, output_path, output_prefix)
    log_memory_usage("init")

    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    segment_library = SegmentLibrary.load_default(genes=chain, organisms=args.species)
    if args.nproc is None:
        args.nproc = max(1, min(8, os.cpu_count() or 1))
    logging.info("Using nproc=%d", args.nproc)

    logging.info("Started loading clonotypes for analysis into MIR object.")
    analysis_repertoire = load_analysis_repertoire(
        input_path,
        segment_library,
        locus,
        args.index_col,
        args.lower_len_cdr3,
        args.higher_len_cdr3,
    )
    logging.info("Analysis repertoire: %s", analysis_repertoire)
    logging.info("Started loading prototypes into MIR object.")
    proto_repertoire = load_prototype_repertoire(
        proto_path,
        segment_library,
        locus,
        args.index_col,
    )
    logging.info("Prototype repertoire: %s", proto_repertoire)

    analysis_repertoire = subsample_repertoire(
        analysis_repertoire,
        args.n_clonotypes,
        args.sample_random_clonotypes,
        args.random_seed,
    )
    proto_repertoire = subsample_repertoire(
        proto_repertoire,
        args.n_prototypes,
        args.sample_random_prototypes,
        args.random_seed,
    )

    clone_representations = get_representations_df(analysis_repertoire, locus)
    metadata_columns = []
    if args.labels_col:
        metadata_columns.append(args.labels_col)
    if args.enrich_by and args.enrich_by not in metadata_columns:
        metadata_columns.append(args.enrich_by)
    metadata = get_metadata_columns(input_path, args.index_col, metadata_columns)
    if metadata is not None:
        clone_representations = clone_representations.merge(metadata, on="clone_id", how="left")
    clone_representations.to_csv(
        f"{output_path}/{output_prefix}_tcremp_representations.tsv",
        sep="\t",
        index=False,
    )

    embeddings = run_tcremp_embedding(
        analysis_repertoire,
        proto_repertoire,
        segment_library,
        chain,
        args.metrics,
        args.nproc,
    )
    logging.info("Finished processing %d clones.", analysis_repertoire.total)

    clone_ids = clone_representations["clone_id"]

    if args.cluster:
        clusters = run_dbscan_clustering(
            embeddings,
            n_components=args.cluster_pc_components,
            min_samples=args.cluster_min_samples,
            n_neighbors=args.k_neighbors,
        )
        cluster_df = pd.DataFrame({"clone_id": clone_ids, "cluster_id": clusters}).merge(clone_representations)
        cluster_df.to_csv(f"{output_path}/{output_prefix}_tcremp_clusters.tsv", sep="\t", index=False)

        if args.enrich_by:
            enrichment_summary = compute_cluster_enrichment(
                cluster_df=cluster_df,
                label_col=args.enrich_by,
                cluster_col="cluster_id",
                threshold=args.enrichment_threshold,
                fdr_threshold=args.enrichment_fdr_threshold,
            )
            cluster_df = annotate_clusters_with_enrichment(
                cluster_df=cluster_df,
                enrichment_summary=enrichment_summary,
                cluster_col="cluster_id",
            )
            save_enrichment_outputs(cluster_df, enrichment_summary, output_prefix, output_path)

    if args.tsne:
        vis_input = embeddings.copy()
        vis_input["clone_id"] = clone_ids.values
        pca_df = pca_proc(vis_input, id_column="clone_id", n_components=args.cluster_pc_components)
        tsne_df = tsne_proc(
            pca_df,
            id_column="clone_id",
            init=args.tsne_init,
            random_state=args.random_seed,
            perplexity=args.tsne_perplexity,
        )

        pca_output = clone_representations.merge(pca_df, on="clone_id", how="inner")
        tsne_output = clone_representations.merge(tsne_df, on="clone_id", how="inner")
        pca_output.to_csv(f"{output_path}/{output_prefix}_tcremp_pca.tsv", sep="\t", index=False)
        tsne_output.to_csv(f"{output_path}/{output_prefix}_tcremp_tsne.tsv", sep="\t", index=False)

        if args.labels_col and args.labels_col in tsne_output.columns:
            labels = [x for x in tsne_output[args.labels_col].dropna().astype(str).unique().tolist()]
            palette = make_custom_palette(labels)
            tsne_plot(
                tsne_output.assign(**{args.labels_col: tsne_output[args.labels_col].fillna("other").astype(str)}),
                to_color=args.labels_col,
                title=f"TCRemP t-SNE: {output_prefix}",
                output_path=f"{output_path}/{output_prefix}_tcremp_tsne.png",
                custom_palette=palette,
            )

    if args.save_dists:
        embeddings["clone_id"] = clone_ids
        embeddings = embeddings[["clone_id"] + [c for c in embeddings.columns if c != "clone_id"]]
        embeddings = clone_representations.merge(embeddings)
        embeddings.to_csv(f"{output_path}/{output_prefix}_tcremp.tsv", sep="\t", index=False)


if __name__ == '__main__':
    main()
