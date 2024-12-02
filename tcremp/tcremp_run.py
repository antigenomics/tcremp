import pandas as pd
import numpy as np
from pathlib import Path, PurePath
import argparse, os, sys
from time import strftime, gmtime
import logging

sys.path.append("../")
sys.path.append("../mirpy/mirpy/")
from tcremp.tcremp_pipeline import TcrempPipeline
from tcremp.tcremp_cluster import TcrempClustering
import tcremp.ml_utils as ml_utils
import tcremp.data_proc as data_proc

tcr_columns = {'TRA': ['a_cdr3aa','a_v','a_j'], 'TRB': ['b_cdr3aa','b_v','b_j'],
               'TRA_TRB': ['a_cdr3aa','a_v','a_j','b_cdr3aa','b_v','b_j']}
tcr_columns_flat = ['cdr3aa','v','j','chain']
clone_label = 'unlabeled'
clone_index_columns = {'TRA': 'cloneId', 'TRB': 'cloneId', 'TRA_TRB': {'TRA':'cloneId_TRA', 'TRB':'cloneId_TRB'}}
species_glossary = {'homosapiens': 'HomoSapiens', 'human': 'HomoSapiens'}
    
def run_clustering(args, tcremp, output_path, output_columns):
    clustering = TcrempClustering(algo_name=args.cluster_algo)
    clustering.build_clusters(chain=args.chain, data=tcremp, label_cl=args.labels_col)
    if args.labels_col:
        df = tcremp.annot[args.chain][output_columns].merge(clustering.clstr_labels[args.chain][['cluster', "label_cluster", tcremp.annotation_id]])
        clustering.clstr_metrics_calc(args.chain, tcremp)
        ##print(f"purity:{model.clstr_metrics[args.chain]['purity']}")
        print(f"retention:{clustering.clstr_metrics[args.chain]['retention']}")
        print(f"f1-score:{clustering.clstr_metrics[args.chain]['f1-score']}")
        print(f"total pairs TCR-epitope:{clustering.clstr_metrics[args.chain]['total pairs TCR-epitope']}")
        print(f"total unique epitopes:{clustering.clstr_metrics[args.chain]['total unique epitopes']}")
        logging.info(f"purity:{clustering.clstr_metrics[args.chain]['purity']}")
        logging.info(f"retention:{clustering.clstr_metrics[args.chain]['retention']}")
        logging.info(f"f1-score:{clustering.clstr_metrics[args.chain]['f1-score']}")
        logging.info(f"total pairs TCR-epitope:{clustering.clstr_metrics[args.chain]['total pairs TCR-epitope']}")
        logging.info(f"total unique epitopes:{clustering.clstr_metrics[args.chain]['total unique epitopes']}")
    else:
        df = tcremp.annot[args.chain][output_columns].merge(clustering.clstr_labels[args.chain][['cluster', tcremp.annotation_id]])
    df.to_csv(f'{output_path}tcremp_clstr_res_{args.chain}.txt', sep='\t', index=False)
    
    

def main():
    parser = argparse.ArgumentParser(description='General TCRemP pipeline implementation')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input file containing a clonotype (clone) table')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output folder')    
    parser.add_argument('-e', '--prefix', type=str,
                        help='Output prefix. Defaults to input clonotype table filename')
    parser.add_argument('-x', '--index-col', type=str,
                        help='(optional) Name of a column in the input table containing user-specified IDs that will be transfered to output tables')
    parser.add_argument('-c', '--chain', type=str, required=True,
                        help='"TRA" or "TRB" for single-chain input data (clonotypes), for paired-chain input (clones) use "TRA_TRB". Used in default prototype set selection')
    parser.add_argument('-p', '--prototypes-path', type=str,
                        help='Path to user-specified folder. If not set, will use default pre-built prototype tables, '
                             '"$tcremp_path/data/data_prebuilt"')
    parser.add_argument('-p_cdr3aa_col', '--prototypes-cdr3aa-column', type=str,
                        help='Name of cdr3 column for custom file with prototypes. "cdr3aa" will be used if not '
                             'specified.')
    parser.add_argument('-p_cdr3nt_col', '--prototypes-cdr3nt-column', type=str,
                        help='Name of cdr3nt column for custom file with prototypes. None will be used if not '
                             'specified.')
    parser.add_argument('-p_v_col', '--prototypes-v-column', type=str,
                        help='Name of v column for custom file with prototypes. "v" will be used if not '
                             'specified.')
    parser.add_argument('-p_j_col', '--prototypes-j-column', type=str,
                        help='Name of j column for custom file with prototypes. "j" will be used if not '
                             'specified.')
    parser.add_argument('-n', '--n-prototypes', type=int,
                        help='Number of prototypes to select for clonotype "triangulation" during embedding. The total number of co-ordinates will be (number of chains) * (3 for V, J and CDR3 distances) * (n). Will use all available prototypes if not set')
    parser.add_argument('-s', '--species', type=str, default='HomoSapiens',
                        help='Prototype set species specification. Currently only "HomoSapiens" is supported')
    parser.add_argument('-u', '--unique-clonotypes',
                        help='Speed-up the analysis by running for unique clonotypes (clones) in the input table')    
    parser.add_argument('-r', '--random-seed', type=int, default=42,
                        help='Random seed for prototype sampling and other rng-based procedures')    
    parser.add_argument('-a', '--cluster-algo', type=str, default='DBSCAN',
                        help='Embedding clustering algorithm: "DBSCAN", "K-means" or "None" to skip the step')    
    parser.add_argument('-l', '--labels-col', type=str,
                        help='(optional) Name of a column in the input table containing clonotype labels. If provided, labels will be transferred to the output and various statistics will be calculated by comparing user-provided labels with inferred cluster labels')
    args = parser.parse_args()

    # IO setup
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    output_prefix = args.prefix
    if not output_prefix:
        output_prefix = input_path.stem

    # Logging
    formatter_str = '[%(asctime)s\t%(name)s\t%(levelname)s] %(message)s'
    formatter = logging.Formatter(formatter_str)
    logging.basicConfig(filename=f'{output_path}/{output_prefix}.log', 
                        format=formatter_str,
                        level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)    
    logging.getLogger().addHandler(handler)
    logging.info(f'Running TCRemP for i="{input_path.resolve()}", writing to o="{output_path.resolve()}/" under prefix="{output_prefix}"')

    # Load input
    if not input_path.is_file:
        logging.error(f'Missing input file "{input_path}"')
        sys.exit('Parameter error')    
    logging.info("Loading data and initializing the pipeline")

    data = pd.read_csv(input_path, sep='\t')

    for col in data.columns:
        data[col] = data[col].astype('str')

    # Check remaining parameters
    species = species_glossary.get(str(args.species).lower())
    if not species:
        logging.error(f'Bad species "{args.species}"')
        sys.exit('Parameter error')
    n_prototypes = args.n_prototypes
    if n_prototypes:
        logging.debug(f'Will use {n_prototypes} prototypes')
        if n_prototypes> 3000:
            logging.warn('More than 3000 prototypes selected, may run very slowly')
    else:
        logging.debug('Will use all available prototypes')
    index_col = args.index_col
    if index_col:
        if index_col in data:
            logging.debug(f'Using {index_col} as clonotype index')
        else:
            logging.error(f'Index column "{index_col}" is missing in input data')
            sys.exit('Bad input')
    label_col = args.labels_col
    if label_col:
        if label_col in data:
            logging.debug(f'Using {label_col} as clonotype labels')
        else:
            logging.error(f'Label column "{label_col}" is missing in input data')
            sys.exit('Bad input')

    # Setup pipeline
    pipeline = TcrempPipeline(run_name=output_path,
                              input_data=data,
                              clonotype_index=index_col,
                              prototypes_path=args.prototypes_path,
                              prototypes_cdr3aa_column=args.prototypes_cdr3aa_column,
                              prototypes_cdr3nt_column=args.prototypes_cdr3nt_column,
                              prototypes_v_column=args.prototypes_v_column,
                              prototypes_j_column=args.prototypes_j_column,
                              n=n_prototypes,
                              species=species,
                              prototypes_chain=args.chain,
                              random_seed=args.random_seed)

    logging.info("Checking input and extracting clonotypes")
    pipeline.tcremp_clonotypes(args.chain, args.unique_clonotypes)
    
    ## output columns
    output_columns = [pipeline.annotation_id, pipeline.clonotype_id] + tcr_columns[args.chain]
    if pipeline.clonotype_index:
        output_columns.append(pipeline.clonotype_index)
    if args.labels_col:
        output_columns.append(args.labels_col)
    
    ## count and save dists
    print('Stage: Distance scores calculation')
    pipeline.tcremp_dists_count(args.chain)
    pipeline.tcremp_dists(args.chain)        
    pipeline.annot[args.chain][output_columns].merge(pipeline.annot_dists[args.chain]).to_csv(f'{output_path}tcremp_dists_{args.chain}.txt', sep='\t', index=False)
    #dist_df = tcremp.annot[args.chain][output_columns].merge(tcremp.annot_dists[args.chain])
    #dist_df.to_csv(f'{output_path}tcremp_dists_{args.chain}.txt', sep='\t', index=False)
    
    ## pca
    print('Stage: PCA calculation')
    pipeline.tcremp_pca(args.chain)
    pipeline.annot[args.chain][output_columns].merge(pipeline.pca[args.chain]).to_csv(f'{output_path}tcremp_pca_{args.chain}.txt', sep='\t', index=False)
    
    ## tsne
    print('Stage: TSNE calculation')
    pipeline.tcremp_tsne(args.chain)
    pipeline.annot[args.chain][output_columns].merge(pipeline.tsne[args.chain]).to_csv(f'{output_path}tcremp_tsne_{args.chain}.txt', sep='\t', index=False)
    
    if args.cluster_algo != 'none':
        logging.info(f'Clustering with algorithm {args.cluster_algo}')
        print(f'Stage: Clustering with algorithm {args.cluster_algo}')
        run_clustering(args, pipeline, output_path, output_columns)
    else:
        print('Finished without clustering')
    
    print(f'Results are in {output_path}')


if __name__ == '__main__':
    main()