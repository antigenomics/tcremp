import sys

sys.path.append("../")

from tcremp.arguments import get_arguments_enrich
from tcremp.utils import configure_logging, load_prototype_repertoire, load_analysis_repertoire, \
    get_representations_df, resolve_prototype_file, \
    resolve_input_file, prepare_output_path, generate_output_prefix, subsample_repertoire
from tcremp.tcremp_run import run_tcremp_embedding
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_cluster import run_dbscan_clustering
import pandas as pd


def main():
    args = get_arguments_enrich()

    input_sample_path = resolve_input_file(args.sample)
    input_background_path = resolve_input_file(args.background)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.sample, args.prefix)

    configure_logging(input_sample_path, output_path, prefix)

    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)

    sample_rep = load_analysis_repertoire(input_sample_path, lib, locus, args.index_col, args.lower_len_cdr3,
                                          args.higher_len_cdr3)
    background_rep = load_analysis_repertoire(input_background_path, lib, locus, args.index_col, args.lower_len_cdr3,
                                              args.higher_len_cdr3)
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)

    sample_rep = subsample_repertoire(sample_rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    background_rep = subsample_repertoire(background_rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)

    sample_emb = run_tcremp_embedding(sample_rep, proto, lib, chain, args.metrics, args.nproc)
    sample_representations = get_representations_df(sample_rep, locus)
    sample_ids = pd.Series([c.id for c in sample_rep])

    background_emb = run_tcremp_embedding(background_rep, proto, lib, chain, args.metrics, args.nproc)
    background_representations = get_representations_df(background_rep, locus)
    background_ids = pd.Series([c.id for c in background_rep])

    joint_embeddings = pd.concat([sample_emb, background_emb])
    joint_representations = pd.concat([sample_representations, background_representations])
    joint_ids = pd.concat([sample_ids, background_ids])
    # TODO drop duplicates
    clust = run_dbscan_clustering(joint_embeddings, args.cluster_pc_components,
                                  args.cluster_min_samples, args.k_neighbors)

    pd.DataFrame({'clone_id': joint_ids, 'cluster_id': clust}).merge(joint_representations).to_csv(
        f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()
