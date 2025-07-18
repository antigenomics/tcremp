import sys
import os
import gc
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path

from pympler import asizeof, muppy, summary as sm

sys.path.append("../")
sys.path.append("../../mirpy")

from tcremp.arguments import get_arguments_enrich
from tcremp.utils import (
    configure_logging, load_prototype_repertoire, load_analysis_repertoire,
    get_representations_df, resolve_prototype_file, resolve_input_file,
    prepare_output_path, generate_output_prefix, subsample_repertoire,
    log_memory_usage, resolve_embedding_file, add_z_binom_pvalues, add_log_fold_change
)
from tcremp.tcremp_run import run_tcremp_embedding
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_cluster import run_dbscan_clustering, prepare_data_for_clustering, get_k_neighbors_distance_matrix, \
    estimate_dbscan_eps


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


def compute_embeddings_if_needed(path, args, is_sample, proto, chain, lib, locus, prefix, output_path):
    tag = 'sample' if is_sample else 'background'
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag)

    if emb_path.exists():
        logging.info(f"Found existing {tag} embeddings at {emb_path}")
        return

    logging.info(f"Computing {tag} embeddings...")
    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    run_tcremp_embedding(rep, proto, lib, chain, args.metrics, args.nproc, emb_path)

    del rep
    logging.info(f"Saved {tag} embeddings to {emb_path}")
    log_memory_usage('Inside the function')


def load_embeddings(path, args, is_sample, lib, locus, prefix, output_path):
    tag = 'sample' if is_sample else 'background'
    prefix_tag = 's_' if is_sample else 'b_'
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag, must_exist=True)
    emb = pd.read_parquet(emb_path)

    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)

    rep_df = get_representations_df(rep, locus)
    rep_df['clone_id'] = prefix_tag + rep_df['clone_id'].astype(str)
    ids = pd.Series([f'{prefix_tag}{c.id}' for c in rep])

    del rep
    gc.collect()
    return emb, rep_df, ids


def compute_cluster_summary(cluster_df, sample_ids):
    sample_ids_set = set(sample_ids)
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


def main():
    args = get_arguments_enrich()

    input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib = setup_environment(
        args)

    log_memory_usage('Init')
    logging.info("Starting TCRempNet pipeline...")

    log_memory_usage("Start. Loading prototypes")
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)

    logging.info("Computing sample embeddings if needed...")
    compute_embeddings_if_needed(input_sample_path, args, is_sample=True, proto=proto, chain=chain, lib=lib,
                                 locus=locus, prefix=prefix, output_path=output_path)
    gc.collect()
    log_memory_usage("After computing sample embeddings")

    logging.info("Computing background embeddings if needed...")
    compute_embeddings_if_needed(input_background_path, args, is_sample=False, proto=proto, chain=chain, lib=lib,
                                 locus=locus, prefix=prefix, output_path=output_path)
    gc.collect()
    log_memory_usage("After computing background embeddings")

    logging.info("Loading sample embeddings...")
    sample_emb, sample_representations, sample_ids = load_embeddings(input_sample_path, args, is_sample=True, lib=lib,
                                                                     locus=locus, prefix=prefix,
                                                                     output_path=output_path)

    logging.info("Loading background embeddings...")
    background_emb, background_representations, background_ids = load_embeddings(input_background_path, args,
                                                                                 is_sample=False, lib=lib, locus=locus,
                                                                                 prefix=prefix, output_path=output_path)

    log_memory_usage("After loading embeddings")

    joint_embeddings = pd.concat([sample_emb, background_emb], ignore_index=True)
    joint_representations = pd.concat([sample_representations, background_representations], ignore_index=True)
    sample_size = len(sample_emb)
    background_size = len(background_emb)
    del sample_emb
    del background_emb

    joint_ids = pd.concat([sample_ids, background_ids], ignore_index=True)
    log_memory_usage("After concatenation")

    logging.info("Running clustering...\n")

    logging.info('Preparing data for clustering...')
    df = prepare_data_for_clustering(joint_embeddings, n_components=args.cluster_pc_components)

    logging.info('Evaluating k-neighbors distance matrix...')
    distances = get_k_neighbors_distance_matrix(df, n_neighbors=args.k_neighbors)

    logging.info('Estimating epsilon for dbscan (by sample embeddings)...')
    eps = estimate_dbscan_eps(df, distances=distances[:, args.k_neighbors - 1])
    # eps_sample = estimate_dbscan_eps(df[:sample_size], distances=distances[:sample_size, args.k_neighbors - 1])
    # eps_background = estimate_dbscan_eps(df[sample_size:], distances=distances[sample_size:, args.k_neighbors - 1])
    # eps = (eps_sample * sample_size + eps_background * background_size) / (sample_size + background_size)
    # logging.info(f'Weighted epsilon is {eps}')

    logging.info("Starting clustering...")
    clust = run_dbscan_clustering(df,
                                  eps=eps,
                                  closest_neigh_dist_array=distances[:, 1],
                                  min_samples=args.cluster_min_samples)
    log_memory_usage("After clustering")

    cluster_df = pd.DataFrame({'clone_id': joint_ids, 'cluster_id': clust})
    cluster_df = cluster_df.merge(joint_representations)
    cluster_df.to_csv(f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)
    logging.info("Saved cluster assignments.")

    summary = compute_cluster_summary(cluster_df, sample_ids)
    summary = add_z_binom_pvalues(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary = add_log_fold_change(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary[['cluster_id', 'cluster_size', 'sample', 'background', 'enrichment_pvalue_zbinom', 'enrichment_fdr_zbinom',
             'log_fold_change']].to_csv(
        f"{output_path}/{prefix}_summary_tcrempnet.tsv", sep='\t', index=False)
    logging.info("Saved cluster summary with p-values.")

    enriched_clusters = summary.loc[
        summary['enrichment_fdr_zbinom'] < 0.05, ['cluster_id', 'enrichment_pvalue_zbinom']
    ]
    logging.info(f"{len(enriched_clusters)} clusters identified as enriched (fdr < 0.05).")

    enriched_clonotypes = cluster_df.merge(enriched_clusters).merge(joint_representations)
    enriched_clonotypes.to_csv(
        f"{output_path}/{prefix}_enriched_clonotypes_tcremp.tsv", sep='\t', index=False
    )
    logging.info("Saved enriched clonotypes.")

    joint_embeddings['clone_id'] = joint_ids
    enriched_embeddings = enriched_clonotypes[
        ['clone_id', 'cluster_id', 'source', 'enrichment_pvalue_zbinom']].merge(joint_embeddings)
    enriched_embeddings.to_csv(
        f"{output_path}/{prefix}_enriched_embeddings_tcremp.tsv", sep='\t', index=False
    )
    logging.info("Saved enriched embeddings.")

    logging.info("TCRempNet pipeline completed.")
    log_memory_usage("Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
