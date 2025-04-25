import sys
sys.path.append("../")

import time
import pandas as pd
import sys
import logging
from pathlib import Path

from mir.common.segments import SegmentLibrary
from mir.common.repertoire import Repertoire
from mir.common.parser import AIRRParser, DoubleChainAIRRParser
from mir.embedding.prototype_embedding import PrototypeEmbedding, Metrics
from mir.distances.aligner import ClonotypeAligner

from tcremp import get_resource_path
from tcremp.arguments import get_arguments
from tcremp.tcremp_cluster import run_dbscan_clustering


def configure_logging(input_path, output_path, output_prefix):
    formatter_str = '[%(asctime)s\t%(name)s\t%(levelname)s] %(message)s'
    formatter = logging.Formatter(formatter_str)
    logging.basicConfig(filename=f'{output_path}/{output_prefix}.log',
                        format=formatter_str,
                        level=logging.DEBUG)  # todo add logging level to cli

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.info(
        f'Running TCRemP for i="{input_path}", writing to o="{output_path.resolve()}/" under prefix="{output_prefix}"')


def configure_io(args):
    # IO setup
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    proto_path = Path(args.prototypes_path if args.prototypes_path else get_resource_path('tcremp_prototypes_olga.tsv'))
    output_prefix = args.prefix
    if not output_prefix:
        output_prefix = Path(input_path).stem
    return str(input_path.resolve()), output_path, output_prefix, str(proto_path.resolve())


def validate_cdr3_len(repertoire: Repertoire, llen: int, hlen: int, single_chain):
    if llen is None:
        llen = -1
    if hlen is None:
        hlen = 35

    def create_len_function(single_chain):
        if single_chain:
            return lambda x: llen <= len(x.cdr3aa) < hlen
        else:
            return lambda x: llen <= len(x.chainA.cdr3aa) < hlen and llen <= len(x.chainB.cdr3aa) < hlen

    logging.info(f'Filtering: {llen} <= cdr3_len < {hlen}')
    len_function = create_len_function(single_chain)

    filtered_repertoire = repertoire.subsample_by_lambda(lambda x: not len_function(x))
    if filtered_repertoire.total > 0:
        for c in filtered_repertoire:
            logging.warning(f'Filtered out clonotype with id {c.id} {c} due to length filter')

    return repertoire.subsample_by_lambda(len_function)


def load_mirpy_objects(segment_library, data_path, proto_path, locus=None,
                       mapping_column=None, llen=None, hlen=None):
    parser = AIRRParser(lib=segment_library,
                        locus=locus) if locus is not None else DoubleChainAIRRParser(lib=segment_library,
                                                                                     mapping_column=mapping_column)
    logging.info('Started loading clonotypes for analysis into MIR object.')
    analysis_repertoire = Repertoire.load(
        parser=parser,
        path=data_path,
    )
    analysis_repertoire = validate_cdr3_len(analysis_repertoire, llen, hlen, locus is not None)
    logging.info('Started loading prototypes into MIR object.')
    proto_repertoire = Repertoire.load(
        parser=parser,
        path=proto_path,
    )
    return analysis_repertoire, proto_repertoire


def validate_sampling_size(rep: Repertoire, n, repertoire_name):
    if n is None:
        return False
    if rep.total < n:
        logging.warning(f'There are less than {n} clonotypes in {repertoire_name} repertoire. Would not perform sampling.')
        return False
    return True


def get_clonotype_representation(clonotype, locus=None):
    def get_one_chain_repr(one_chain_clone):
        return '_'.join([one_chain_clone.cdr3aa, one_chain_clone.v.id, one_chain_clone.j.id])

    if locus is None:
        return '/'.join([get_one_chain_repr(clonotype.chainA), get_one_chain_repr(clonotype.chainB)])
    else:
        return get_one_chain_repr(clonotype)


def main():
    args = get_arguments()
    input_path, output_path, output_prefix, proto_path = configure_io(args)
    configure_logging(input_path, output_path, output_prefix)

    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    segment_library = SegmentLibrary.load_default(genes=chain,
                                                  organisms=args.species)

    analysis_repertoire, proto_repertoire = load_mirpy_objects(segment_library=segment_library,
                                                               data_path=input_path,
                                                               proto_path=proto_path,
                                                               locus=locus,
                                                               mapping_column=args.index_col,
                                                               llen=args.lower_len_cdr3,
                                                               hlen=args.higher_len_cdr3)

    if validate_sampling_size(analysis_repertoire, args.n_clonotypes, 'analysis'):
        analysis_repertoire = analysis_repertoire.sample_n(n=args.n_clonotypes,
                                                           sample_random=args.sample_random_prototypes,
                                                           random_seed=args.random_seed)
    if validate_sampling_size(proto_repertoire, args.n_prototypes, 'prototypes'):
        proto_repertoire = proto_repertoire.sample_n(n=args.n_prototypes,
                                                     sample_random=args.sample_random_clonotypes,
                                                     random_seed=args.random_seed)

    logging.info('Processed input clonotype and prototype data.')
    logging.info(f'There are {analysis_repertoire.total} analysis clonotypes and {proto_repertoire.total} prototypes.')
    logging.info('Initializing aligner.')
    t0 = time.time()
    aligner = ClonotypeAligner.from_library(lib=segment_library)
    logging.info(f'Initialized aligner object in {time.time() - t0}.')

    embedding_maker = PrototypeEmbedding(proto_repertoire,
                                         aligner=aligner,
                                         metrics=Metrics(args.metrics),
                                         )
    logging.info(f'Running the embedding calculation.')

    t0 = time.time()
    embeddings = embedding_maker.embed_repertoire(analysis_repertoire,
                                                  threads=args.nproc,
                                                  flatten_scores=True)
    logging.info(f'Embeddings have been evaluated')
    column_names = []
    for i in range(proto_repertoire.total):
        if 'TRA' in chain:
            column_names += [f'{i}_a_v', f'{i}_a_j', f'{i}_a_cdr3']
        if 'TRB' in chain:
            column_names += [f'{i}_b_v', f'{i}_b_j', f'{i}_b_cdr3']
    embeddings = pd.DataFrame(embeddings, columns=column_names)
    logging.info(f'Finished {analysis_repertoire.total} clones in {time.time() - t0}')
    clone_ids = [get_clonotype_representation(c, locus) for c in analysis_repertoire]

    if args.cluster:
        clusters = run_dbscan_clustering(embeddings,
                                         n_components=args.cluster_pc_components,
                                         min_samples=args.cluster_min_samples)
        cluster_df = pd.DataFrame({'clone_id': clone_ids,
                                   'cluster_id': clusters})
        cluster_df.to_csv(f'{output_path}/{output_prefix}_tcremp_clusters.tsv', sep='\t', index=False)

    embeddings['clone_id'] = clone_ids
    embeddings = embeddings[['clone_id'] + column_names]
    if args.save_dists:
        embeddings.to_csv(f'{output_path}/{output_prefix}_tcremp.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
