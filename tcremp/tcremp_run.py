import sys

sys.path.append("../")

from tcremp.arguments import get_arguments
from tcremp.utils import configure_logging, load_prototype_repertoire, load_analysis_repertoire, \
    get_representations_df, resolve_prototype_file, \
    resolve_input_file, prepare_output_path, generate_output_prefix, subsample_repertoire
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_cluster import run_dbscan_clustering
import pandas as pd
import time
import logging
from mir.embedding.prototype_embedding import PrototypeEmbedding, Metrics
from mir.distances.aligner import ClonotypeAligner


def run_tcremp_embedding(analysis_rep, proto_rep, segment_library, chain, metrics, nproc):
    aligner = ClonotypeAligner.from_library(lib=segment_library)
    embedder = PrototypeEmbedding(proto_rep, aligner=aligner, metrics=Metrics(metrics))
    t0 = time.time()
    emb = embedder.embed_repertoire(analysis_rep, threads=nproc, flatten_scores=True)
    logging.info(f'Embeddings done in {time.time() - t0:.2f}s')

    columns = []
    for i in range(proto_rep.total):
        if 'TRA' in chain:
            columns += [f'{i}_a_v', f'{i}_a_j', f'{i}_a_cdr3']
        if 'TRB' in chain:
            columns += [f'{i}_b_v', f'{i}_b_j', f'{i}_b_cdr3']
    return pd.DataFrame(emb, columns=columns)


def main():
    args = get_arguments()

    input_path = resolve_input_file(args.input)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.input, args.prefix)

    configure_logging(input_path, output_path, prefix)

    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)

    rep = load_analysis_repertoire(input_path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)

    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)

    emb = run_tcremp_embedding(rep, proto, lib, chain, args.metrics, args.nproc)
    reps = get_representations_df(rep, locus)
    ids = pd.Series([c.id for c in rep])

    if args.cluster:
        clust = run_dbscan_clustering(emb, args.cluster_pc_components,
                                      args.cluster_min_samples, args.k_neighbors)
        pd.DataFrame({'clone_id': ids, 'cluster_id': clust}).merge(reps).to_csv(
            f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)

    if args.save_dists:
        emb['clone_id'] = ids
        emb = emb[['clone_id'] + [c for c in emb.columns if c != 'clone_id']]
        emb = reps.merge(emb)
        emb.to_csv(f"{output_path}/{prefix}_tcremp.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()
