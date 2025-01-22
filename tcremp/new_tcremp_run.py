import time
import pandas as pd

from tcremp import get_resource_path
from tcremp.arguments import get_arguments
import sys
import logging
from pathlib import Path

# sys.path.append('../../mirpy')

from mir.common.segments import SegmentLibrary
from mir.common.repertoire import Repertoire
from mir.common.parser import AIRRParser, DoubleChainAIRRParser
from mir.embedding.prototype_embedding import PrototypeEmbedding
from mir.distances.aligner import ClonotypeAligner


def validate_input_args(args):
    chain_options = ['TRA', 'TRB', 'TRA_TRB']
    if args.chain not in chain_options:
        raise KeyError(f'Chain must be one of: {chain_options}')

    species_options = ['HomoSapiens', 'MusMusculus', 'MaccaMulatta']
    if args.species not in species_options:
        raise KeyError(f'Species must be one of: {species_options}')


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


def main(args):
    validate_input_args(args)
    input_path, output_path, output_prefix, proto_path = configure_io(args)
    configure_logging(input_path, output_path, output_prefix)

    chain = args.chain.split('_')
    single_chain = len(chain) == 1
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

    analysis_repertoire = analysis_repertoire.sample_n(n=args.n_clonotypes,
                                                       sample_random=args.sample_random_prototypes,
                                                       random_seed=args.random_seed)
    proto_repertoire = proto_repertoire.sample_n(n=args.n_prototypes,
                                                 sample_random=args.sample_random_clonotypes,
                                                 random_seed=args.random_seed)

    logging.info('Processed input clonotype and prototype data. ')
    logging.info(f'There are {analysis_repertoire.total} analysis clonotypes and {proto_repertoire.total} prototypes.')
    logging.info('Initializing aligner.')
    t0 = time.time()
    aligner = ClonotypeAligner.from_library(lib=segment_library)
    logging.info(f'Initialized aligner object in {time.time() - t0}. Initializing embedding object.')

    embedding_maker = PrototypeEmbedding(proto_repertoire,
                                         aligner=aligner
                                         )
    logging.info(f'Running the embedding calculation.')

    t0 = time.time()
    embeddings = embedding_maker.embed_repertoire(analysis_repertoire,
                                                  threads=args.nproc,
                                                  flatten_scores=True)
    column_names = []
    for i in range(proto_repertoire.total):
        if 'TRA' in chain:
            column_names += [f'{i}_a_v', f'{i}_a_j', f'{i}_a_cdr3']
        if 'TRB' in chain:
            column_names += [f'{i}_b_v', f'{i}_b_j', f'{i}_b_cdr3']
    embeddings = pd.DataFrame(embeddings, columns=column_names)

    logging.info(f'Finished {analysis_repertoire.total} clones in {time.time() - t0}')
    embeddings['clone_id'] = [x.id for x in analysis_repertoire]
    embeddings = embeddings[['clone_id'] + column_names]
    embeddings.to_csv(f'{output_path}/{output_prefix}_tcremp.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main(get_arguments())
