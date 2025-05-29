import sys

sys.path.append("../")

from tcremp.arguments import get_arguments_enrich
from tcremp.utils import configure_logging, load_prototype_repertoire, load_analysis_repertoire, \
    get_representations_df, resolve_prototype_file, \
    resolve_input_file, prepare_output_path, generate_output_prefix, subsample_repertoire, add_fisher_pvalues
from tcremp.tcremp_run import run_tcremp_embedding
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_cluster import run_dbscan_clustering
import pandas as pd

import logging


def main():
    args = get_arguments_enrich()

    input_sample_path = resolve_input_file(args.sample)
    input_background_path = resolve_input_file(args.background)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.sample, args.prefix)

    configure_logging(input_sample_path, output_path, prefix)
    logging.info("Starting TCRempNet pipeline...")

    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)

    logging.info("Loading repertoires...")
    sample_rep = load_analysis_repertoire(input_sample_path, lib, locus, args.index_col, args.lower_len_cdr3,
                                          args.higher_len_cdr3)
    background_rep = load_analysis_repertoire(input_background_path, lib, locus, args.index_col, args.lower_len_cdr3,
                                              args.higher_len_cdr3)
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)

    sample_rep = subsample_repertoire(sample_rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    background_rep = subsample_repertoire(background_rep, args.n_clonotypes, args.sample_random_prototypes,
                                          args.random_seed)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)

    logging.info("Computing embeddings...")
    sample_emb = run_tcremp_embedding(sample_rep, proto, lib, chain, args.metrics, args.nproc)
    background_emb = run_tcremp_embedding(background_rep, proto, lib, chain, args.metrics, args.nproc)

    sample_representations = get_representations_df(sample_rep, locus)
    background_representations = get_representations_df(background_rep, locus)
    sample_representations['clone_id'] = 's_' + sample_representations['clone_id'].astype(str)
    background_representations['clone_id'] = 'b_' + background_representations['clone_id'].astype(str)
    sample_ids = pd.Series([f's_{c.id}' for c in sample_rep])
    background_ids = pd.Series([f'b_{c.id}' for c in background_rep])

    joint_embeddings = pd.concat([sample_emb, background_emb])
    joint_representations = pd.concat([sample_representations, background_representations])
    joint_ids = pd.concat([sample_ids, background_ids])

    logging.info("Running clustering...")
    clust = run_dbscan_clustering(joint_embeddings, args.cluster_pc_components,
                                  args.cluster_min_samples, args.k_neighbors)

    cluster_df = pd.DataFrame({'clone_id': joint_ids, 'cluster_id': clust})
    cluster_df = cluster_df.merge(joint_representations)
    cluster_df.to_csv(f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)
    logging.info("Saved cluster assignments.")

    sample_ids_set = set(sample_ids)
    background_ids_set = set(background_ids)
    cluster_df['source'] = cluster_df['clone_id'].apply(
        lambda x: 'sample' if x in sample_ids_set else 'background'
    )

    logging.info("Computing cluster summary...")
    summary = (
        cluster_df.groupby('cluster_id')['source']
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis(index='cluster_id', columns=None)
        .reset_index()
    )
    summary['cluster_size'] = summary.get('sample', 0) + summary.get('background', 0)
    summary = summary[summary.cluster_id != -1]

    summary = add_fisher_pvalues(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary[['cluster_id', 'cluster_size', 'sample', 'background', 'enrichment_pvalue']].to_csv(
        f"{output_path}/{prefix}_summary_tcrempnet.tsv", sep='\t', index=False
    )
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


if __name__ == "__main__":
    main()
