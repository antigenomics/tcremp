import sys
import os
import gc
import logging
import multiprocessing as mp
import pandas as pd

sys.path.append("../")

from tcremp.arguments import get_arguments_enrich
from tcremp.utils import (
    configure_logging, load_prototype_repertoire, load_analysis_repertoire,
    get_representations_df, resolve_prototype_file, resolve_input_file,
    prepare_output_path, generate_output_prefix, subsample_repertoire,
    log_memory_usage, add_fisher_pvalues
)
from tcremp.tcremp_run import run_tcremp_embedding
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_cluster import run_dbscan_clustering


def setup_environment(args):
    input_sample_path = resolve_input_file(args.sample)
    input_background_path = resolve_input_file(args.background)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.sample, args.prefix)

    configure_logging(input_sample_path, output_path, prefix)
    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)

    return input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib


def process_repertoire(path, lib, locus, args, is_sample, proto, chain):
    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    emb = run_tcremp_embedding(rep, proto, lib, chain, args.metrics, args.nproc)
    rep_df = get_representations_df(rep, locus)
    prefix = 's_' if is_sample else 'b_'
    rep_df['clone_id'] = prefix + rep_df['clone_id'].astype(str)
    ids = pd.Series([f'{prefix}{c.id}' for c in rep])
    del rep
    gc.collect()
    return emb, rep_df, ids


def load_temp_embeddings(path):
    return pd.read_parquet(path)


def compute_cluster_summary(cluster_df, sample_ids, background_ids):
    sample_ids_set = set(sample_ids)
    background_ids_set = set(background_ids)
    cluster_df['source'] = cluster_df['clone_id'].apply(
        lambda x: 'sample' if x in sample_ids_set else 'background')

    summary = (
        cluster_df.groupby('cluster_id')['source']
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis(index='cluster_id', columns=None)
        .reset_index()
    )
    summary['cluster_size'] = summary.get('sample', 0) + summary.get('background', 0)
    summary = summary[summary.cluster_id != -1]
    return summary


def load_embeddings_or_compute(path, args, is_sample, proto, chain, lib, locus, prefix, output_path):
    from pathlib import Path
    if is_sample:
        logging.info("Loading sample repertoires...")
        emb_path = args.sample_embeddings
        prefix_tag = 's_'
    else:
        logging.info("Loading background repertoire...")
        emb_path = args.background_embeddings
        prefix_tag = 'b_'

    default_emb_path = Path(output_path) / f"{prefix}_{'sample' if is_sample else 'background'}_embeddings.parquet"
    if emb_path or default_emb_path.exists():
        if not emb_path:
            emb_path = default_emb_path
        emb = load_temp_embeddings(emb_path)
        rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
        rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
        rep_df = get_representations_df(rep, locus)
        rep_df['clone_id'] = prefix_tag + rep_df['clone_id'].astype(str)
        ids = pd.Series([f'{prefix_tag}{c.id}' for c in rep])
        del rep
        gc.collect()
    else:
        emb, rep_df, ids = process_repertoire(path, lib, locus, args, is_sample=is_sample, proto=proto, chain=chain)
        emb.to_parquet(f"{output_path}/{prefix}_{'sample' if is_sample else 'background'}_embeddings.parquet",
                       index=False)
        logging.info(f"Saved {'sample' if is_sample else 'background'} embeddings to file.")
    return emb, rep_df, ids


def main():
    args = get_arguments_enrich()

    input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib = setup_environment(
        args)

    log_memory_usage('Init')
    logging.info("Starting TCRempNet pipeline...")

    log_memory_usage("Start. Loading prototypes")
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)

    logging.info("Loading sample repertoires...")
    sample_emb, sample_representations, sample_ids = load_embeddings_or_compute(
        input_sample_path, args, is_sample=True, proto=proto, chain=chain, lib=lib, locus=locus,
        prefix=prefix, output_path=output_path
    )

    log_memory_usage("After sample embeddings")

    logging.info("Loading background repertoire...")
    background_emb, background_representations, background_ids = load_embeddings_or_compute(
        input_background_path, args, is_sample=False, proto=proto, chain=chain, lib=lib, locus=locus,
        prefix=prefix, output_path=output_path
    )

    log_memory_usage("After background embedding")

    log_memory_usage('After representations')
    joint_embeddings = pd.concat([sample_emb, background_emb], ignore_index=True)
    joint_representations = pd.concat([sample_representations, background_representations], ignore_index=True)
    joint_ids = pd.concat([sample_ids, background_ids], ignore_index=True)
    log_memory_usage("After concatenation")

    logging.info("Running clustering...")
    clust = run_dbscan_clustering(joint_embeddings, args.cluster_pc_components, args.cluster_min_samples,
                                  args.k_neighbors)
    log_memory_usage("After clustering")

    cluster_df = pd.DataFrame({'clone_id': joint_ids, 'cluster_id': clust})
    cluster_df = cluster_df.merge(joint_representations)
    cluster_df.to_csv(f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)
    logging.info("Saved cluster assignments.")

    summary = compute_cluster_summary(cluster_df, sample_ids, background_ids)
    summary = add_fisher_pvalues(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary[['cluster_id', 'cluster_size', 'sample', 'background', 'enrichment_pvalue']].to_csv(
        f"{output_path}/{prefix}_summary_tcrempnet.tsv", sep='\t', index=False)
    logging.info("Saved cluster summary with p-values.")

    enriched_clusters = summary.loc[
        summary['enrichment_pvalue'] < 0.05, ['cluster_id', 'enrichment_pvalue']
    ]
    logging.info(f"{len(enriched_clusters)} clusters identified as enriched (pval < 0.05).")

    enriched_clonotypes = cluster_df.merge(enriched_clusters).merge(joint_representations)
    enriched_clonotypes.to_csv(
        f"{output_path}/{prefix}_enriched_clonotypes_tcremp.tsv", sep='\t', index=False
    )
    logging.info("Saved enriched clonotypes.")

    joint_embeddings['clone_id'] = joint_ids
    enriched_embeddings = enriched_clonotypes[
        ['clone_id', 'cluster_id', 'source', 'enrichment_pvalue']].merge(joint_embeddings)
    enriched_embeddings.to_csv(
        f"{output_path}/{prefix}_enriched_embeddings_tcremp.tsv", sep='\t', index=False
    )
    logging.info("Saved enriched embeddings.")

    logging.info("TCRempNet pipeline completed.")
    log_memory_usage("Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
