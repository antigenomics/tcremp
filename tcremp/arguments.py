import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='General TCRemP pipeline implementation')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input file containing a clonotype (clone) table.')

    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output folder.')

    parser.add_argument('-e', '--prefix', type=str,
                        help='Output prefix. Defaults to input clonotype table filename.')

    parser.add_argument('-x', '--index-col', type=str,
                        help='(optional) Name of a column in the input table containing user-specified IDs that will '
                             'be transfered to output tables.')  # todo missing in readme ++

    parser.add_argument('-c', '--chain', type=str, required=True,
                        choices=['TRA', 'TRB', 'TRA_TRB'],
                        help='"TRA" or "TRB" for single-chain input data (clonotypes), for paired-chain input ('
                             'clones) use "TRA_TRB". Used in default prototype set selection.')

    parser.add_argument('-p', '--prototypes-path', type=str,
                        help='Path to user-specified prototypes file. If not set, will use pre-built prototype tables, '
                             '"$tcremp_path/data/data_prebuilt".')

    parser.add_argument('-n', '--n-prototypes', type=int,
                        help='Number of prototypes to select for clonotype "triangulation" during embedding. The '
                             'total number of co-ordinates will be (number of chains) * (3 for V, J and CDR3 '
                             'distances) * (n). Will use all available prototypes if not set.')

    parser.add_argument('-sample_random_p', '--sample-random-prototypes', type=bool, default=False,
                        help='Whether to sample the prototypes randomly or not. Defaults to False.')

    parser.add_argument('-nc', '--n-clonotypes', type=int,
                        help='Number of clonotypes to process in the pipeline. Will use all available clonotypes'
                             ' if not set.')

    parser.add_argument('-sample_random_c', '--sample-random-clonotypes', type=bool, default=False,
                        help='Whether to sample the clonotypes randomly or not. Defaults to False.')

    parser.add_argument('-s', '--species', type=str, default='HomoSapiens',
                        choices=['HomoSapiens', 'MusMusculus', 'MacacaMulatta'],
                        help='V/J gene aligner species specification. Defaults to HomoSapiens.')

    # parser.add_argument('-u', '--unique-clonotypes',
    #                     help='Speed-up the analysis by running for unique clonotypes (clones) in the input table')  # +

    parser.add_argument('-r', '--random-seed', type=int, default=42,
                        help='Random seed for prototype sampling and other rng-based procedures. Defaults to 42.')

    parser.add_argument('-np', '--nproc', type=int, default=1,
                        help='Number of processes to perform calculation with. Will use 1 process by default.')

    parser.add_argument('-llen', '--lower-len-cdr3', type=int, default=5,
                        help='Filter out cdr3 with len <llen. Defaults to 5.')

    parser.add_argument('-hlen', '--higher-len-cdr3', type=int, default=30,
                        help='Filter out cdr3 with len >=hlen. Defaults to 30.')
    return parser.parse_args()
