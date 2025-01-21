import time
import pandas as pd

from tcremp.arguments import get_arguments
import sys
import logging
from pathlib import Path

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
    proto_path = Path(args.prototypes_path)
    output_prefix = args.prefix
    if not output_prefix:
        output_prefix = Path(input_path).stem
    return str(input_path.resolve()), output_path, output_prefix, str(proto_path.resolve())


def load_mirpy_objects(segment_library, data_path, proto_path, single_chain=True):
    parser = AIRRParser if single_chain else DoubleChainAIRRParser
    logging.info('Started loading clonotypes for analysis into MIR object.')
    analysis_repertoire = Repertoire.load(
        parser=parser(lib=segment_library),
        path=data_path
    )
    logging.info('Started loading prototypes into MIR object.')
    proto_repertoire = Repertoire.load(
        parser=parser(lib=segment_library),
        path=proto_path
    )
    return analysis_repertoire, proto_repertoire


def main(args):
    validate_input_args(args)
    input_path, output_path, output_prefix, proto_path = configure_io(args)
    configure_logging(input_path, output_path, output_prefix)

    chain = args.chain.split('_')
    single_chain = len(chain) == 1
    segment_library = SegmentLibrary.load_default(genes=chain,
                                                  organisms=args.species)

    analysis_repertoire, proto_repertoire = load_mirpy_objects(segment_library, input_path,
                                                               proto_path, single_chain)
    # todo prototypes might be single chain in this case???

    logging.info('Processed input clonotype and prototype data. Start calculating embedding.')

    embedding_maker = PrototypeEmbedding(proto_repertoire,
                                         aligner=ClonotypeAligner.from_library(
                                             lib=segment_library)
                                         )
    t0 = time.time()
    embeddings = embedding_maker.embed_repertoire(analysis_repertoire,
                                                  threads=args.nproc,
                                                  flatten_scores=True)
    column_names = []
    for i in range(proto_repertoire.total):
        column_names += [f'{i}_a_v', f'{i}_a_j', f'{i}_a_cdr3', f'{i}_b_v', f'{i}_b_j', f'{i}_b_cdr3']
    embeddings = pd.DataFrame(embeddings, columns=column_names)

    logging.info(f'Finished {analysis_repertoire.total} clones in {time.time() - t0}')
    embeddings['clone_id'] = [x.id for x in analysis_repertoire]
    embeddings = embeddings[['clone_id'] + column_names]
    embeddings.to_csv(f'{output_path}/{output_prefix}_tcremp.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main(get_arguments())
