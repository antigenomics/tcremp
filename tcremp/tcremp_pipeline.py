import sys, os, time, logging, warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import tcremp.data_proc as data_proc
import tcremp.ml_utils as ml_utils

sys.path.append("../mirpy/")
from mir.common.repertoire import Repertoire
from mir.common.segments import SegmentLibrary
from mir.common import parser
from mir.distances import ClonotypeAligner, GermlineAligner
from mir.comparative import DenseMatcher
from tcremp import get_resource_path


class TcrempPipeline:
    clonotype_id = 'cloneId'
    annotation_id = 'annotId'
    random_state = 7

    def __init__(self, run_name, input_data, clonotype_index=None, prototypes_path=None,
                 prototypes_cdr3aa_column=None, prototypes_cdr3nt_column=None, prototypes_v_column=None,
                 prototypes_j_column=None, n=None, species='HomoSapiens',
                 lower_len_cdr3=5, higher_len_cdr3=30,
                 prototypes_chain='TRA_TRB', random_seed=None):
        self.__prototypes_path_subsets = {'HomoSapiens': {'TRA': get_resource_path('olga_humanTRA.txt'),
                                                          'TRB': get_resource_path('olga_humanTRB.txt')}}
        # self.segments_path = 'data/segments.txt'
        self.segments_path = get_resource_path('segments.txt')
        # self.run_name = run_name
        self.species = species
        self.clonotypes = {}  ## extracted clonotypes
        # self.clonotype_label_pairs = {}
        self.annot_input = {}  ## raw input
        self.annot = {}  ## processed input table (cleaned clonotypes, added annotation id and clonotype id)
        self.dists = {}  ## dists data for clonotypes with clonotype id
        self.annot_dists = {}  ## annotation id with dists data
        self.pca_clones = {}
        self.pca = {}
        self.pca_clone_label = {}
        self.pca_ad = {'clones': {}, 'all': {}}
        self.tsne = {}
        self.tsne_clones = {}
        self.clstr_labels = {}
        self.clsf_labels = {}

        self.tcr_columns = ['cdr3aa', 'v', 'j', 'chain']
        # self.tcr_columns_paired = {'TRA':['a_cdr3aa','TRAV','TRAJ'],'TRB':['b_cdr3aa','TRBV','TRBJ']}
        self.tcr_columns_paired = {'TRA': ['a_cdr3aa', 'a_v', 'a_j'], 'TRB': ['b_cdr3aa', 'b_v', 'b_j']}
        # self.__rename_tcr_columns_paired = {'TRA':{'a_cdr3aa':'cdr3aa','TRAV':'v','TRAJ':'j','cloneId_TRA':'cloneId'},'TRB':{'b_cdr3aa':'cdr3aa','TRBV':'v','TRBJ':'j','cloneId_TRB':'cloneId'}}
        self.__rename_tcr_columns_paired = {
            'TRA': {'a_cdr3aa': 'cdr3aa', 'a_v': 'v', 'a_j': 'j', 'cloneId_TRA': 'cloneId'},
            'TRB': {'b_cdr3aa': 'cdr3aa', 'b_v': 'v', 'b_j': 'j', 'cloneId_TRB': 'cloneId'}}
        self.clonotype_id = 'cloneId'
        self.clonotyoe_label_id = 'pairId'
        self.input_id = 'inputId'
        # self.annotation_id = 'annotId'
        # self.annotation_id = 'id' ## index
        self.annotation_id = 'tcremp_id'  ## index
        self.clonotype_id_dict = {'TRA': 'cloneId', 'TRB': 'cloneId',
                                  'TRA_TRB': {'TRA': 'cloneId_TRA', 'TRB': 'cloneId_TRB'}}

        self.prototypes_path = self.__prototypes_path_subsets[species]
        # self.__prototypes_path = prototypes_path

        self.__n_components = 50

        self.__tsne_init = 'pca'
        self.__tsne_perplexity = 15
        self.__random_state = 7
        self.time_dict = {}
        self.prototypes_cdr3aa_column = prototypes_cdr3aa_column
        self.prototypes_cdr3nt_column = prototypes_cdr3nt_column
        self.prototypes_v_column = prototypes_v_column
        self.prototypes_j_column = prototypes_j_column
        self.lower_len_cdr3 = lower_len_cdr3
        self.higher_len_cdr3 = higher_len_cdr3

        self.outputs_path = os.path.join(run_name, '')
        Path(self.outputs_path).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=f'{self.outputs_path}tcremp_log.log', level=logging.DEBUG)

        os.remove(f'{self.outputs_path}filtered_out_data.txt') if os.path.exists(
            f'{self.outputs_path}filtered_out_data.txt') else print('')

        self.clonotypes_path = {'TRA': self.outputs_path + 'clonotypes_TRA.txt',
                                'TRB': self.outputs_path + 'clonotypes_TRB.txt',
                                'TRA_TRB': {'TRA': self.outputs_path + 'clonotypes_paired_TRA.txt',
                                            'TRB': self.outputs_path + 'clonotypes_paired_TRB.txt'}}
        self.dists_res_path = {'TRA': self.outputs_path + 'res_TRA.txt', 'TRB': self.outputs_path + 'res_TRB.txt',
                               'TRA_TRB': {'TRA': self.outputs_path + 'res_paired_TRA.txt',
                                           'TRB': self.outputs_path + 'res_paired_TRB.txt'}}

        self.clonotype_index = clonotype_index
        self.raw_input_data = input_data.copy()
        self.check_proc_input_data()
        self.input_data = self.__annot_id(self.input_data, self.input_id)

        if prototypes_path:
            self.prototypes_path = {'TRA': self.outputs_path + 'prototypes_TRA.txt',
                                    'TRB': self.outputs_path + 'prototypes_TRB.txt'}
            self.prototypes_prep(prototypes_path,
                                 cdr3aa_column=self.prototypes_cdr3aa_column,
                                 cdr3nt_column=self.prototypes_cdr3nt_column,
                                 v_column=self.prototypes_v_column,
                                 j_column=self.prototypes_j_column)

        if n:
            new_prototypes_path = {'TRA': self.outputs_path + f'prototypes_TRA_{n}.txt',
                                   'TRB': self.outputs_path + f'prototypes_TRB_{n}.txt'}
            # print(new_prototypes_path)
            try:
                if prototypes_chain == 'TRA_TRB':
                    self.process_prototypes_file(n, self.prototypes_path['TRA'], new_prototypes_path['TRA'],
                                                 random_seed=random_seed)
                    self.process_prototypes_file(n, self.prototypes_path['TRB'], new_prototypes_path['TRB'],
                                                 random_seed=random_seed)
                else:
                    self.process_prototypes_file(n, self.prototypes_path[prototypes_chain],
                                                 new_prototypes_path[prototypes_chain],
                                                 random_seed=random_seed)
            except ValueError:
                print('n is greater than number of clonotypes in prototypes file')
                logging.error('n is greater than number of clonotypes in prototypes file')
            self.prototypes_path = new_prototypes_path
            if n * 2 < 50:
                self.__n_components = n * 2
        else:
            n = 3000

        self.dist_cols_dist = {'TRA': [f'a_{xs}_{x}' for xs in range(n) for x in ['v', 'j', 'cdr3']],
                               'TRB': [f'b_{xs}_{x}' for xs in range(n) for x in ['v', 'j', 'cdr3']]}

    def check_proc_input_data(self):
        data_proc.check_columns(self.raw_input_data, self.tcr_columns_paired['TRA'] + self.tcr_columns_paired['TRB'])
        self.input_data = data_proc.clean_at_least_cdr3a_or_cdr3b(self.raw_input_data,
                                                                  self.tcr_columns_paired['TRA'][0],
                                                                  self.tcr_columns_paired['TRB'][0], self.outputs_path)
        for current_chain in ['TRA', 'TRB']:
            self.input_data = data_proc.remove_asterisk(self.input_data, self.tcr_columns_paired[current_chain])
            self.input_data = data_proc.remove_backslash(self.input_data, self.tcr_columns_paired[current_chain])
            self.input_data = data_proc.add_allele(self.input_data, self.tcr_columns_paired[current_chain])

    def prototypes_prep(self, input_file_path, cdr3aa_column, cdr3nt_column, v_column, j_column):
        if cdr3aa_column is None:
            cdr3aa_column = 'cdr3aa'
        if v_column is None:
            v_column = 'v'
        if j_column is None:
            j_column = 'j'
        prototypes = pd.read_csv(input_file_path, sep='\t')
        prototypes = data_proc.filter_clones_data(prototypes, [cdr3aa_column, v_column, j_column],
                                                  cdr3nt=cdr3nt_column)
        prototypes = data_proc.filter_segments(prototypes, segments_path=self.segments_path, v=v_column, j=j_column)

        prototypes_a = prototypes[prototypes['chain'] == 'TRA']
        if len(prototypes_a) > 0:
            prototypes_a.reset_index(drop=True).drop('chain', axis=1).to_csv(self.prototypes_path['TRA'], sep='\t')

        prototypes_b = prototypes[prototypes['chain'] == 'TRB']
        if len(prototypes_b) > 0:
            prototypes_b.reset_index(drop=True).drop('chain', axis=1).to_csv(self.prototypes_path['TRB'], sep='\t')

    def process_prototypes_file(self, n, old_path, new_path, random_seed=None):
        prototypes_data = pd.read_csv(old_path, sep='\t', index_col=0)
        if random_seed is not None and len(prototypes_data) != n:
            prototypes_data = prototypes_data.sample(n=n, random_state=random_seed)
        else:
            prototypes_data = prototypes_data.iloc[:n]
        prototypes_data.rename(
            columns={self.prototypes_v_column: 'v',
                     self.prototypes_j_column: 'j',
                     self.prototypes_cdr3aa_column: 'cdr3aa'}).reset_index(
            drop=True).to_csv(new_path, sep='\t')

    def __annot_id(self, data, annotation_id_str):
        df = data.copy()
        df[annotation_id_str] = df.index
        return df

    def __assign_clones_ids(self, data, chain):
        df = data.copy()
        df[self.clonotype_id] = df.groupby(self.tcr_columns_paired[chain], dropna=False).ngroup()
        return df

    def __assign_clones_ids_paired(self, data, chain):
        df = data.copy()
        if chain == 'TRA' or chain == 'TRB':
            df[self.clonotype_id_dict['TRA_TRB'][chain]] = df.groupby(self.tcr_columns_paired[chain],
                                                                      dropna=False).ngroup()
        if chain == 'TRA_TRB':
            df[self.clonotype_id] = df.groupby(self.tcr_columns_paired['TRA'] + self.tcr_columns_paired['TRB'],
                                               dropna=False).ngroup()
        return df

    def __clonotypes_prep_old(self, clones_df, chain, tcr_columns, clonotype_id_str):
        clonotypes = clones_df[clones_df['chain'] == chain]
        clonotypes = clones_df.copy()
        clonotypes = data_proc.remove_asterisk(clonotypes, tcr_columns)
        clonotypes = data_proc.remove_backslash(clonotypes, tcr_columns)
        clonotypes = data_proc.filter_clones_data(clonotypes, tcr_columns)
        clonotypes = data_proc.filter_segments(clonotypes, segments_path='../mirpy/mirpy/mir/resources/segments.txt',
                                               organism=self.species)

        clonotypes = clonotypes[tcr_columns + [clonotype_id_str]].drop_duplicates().reset_index(drop=True)
        return clonotypes

    def __clonotypes_prep(self, clones_df, chain):
        clonotypes = clones_df.copy()
        clonotypes = clonotypes.rename(self.__rename_tcr_columns_paired[chain], axis=1)
        clonotypes['chain'] = chain
        clonotypes = clonotypes[self.tcr_columns + [self.clonotype_id]].drop_duplicates().reset_index(drop=True)
        clonotypes['d'] = '.'
        return clonotypes

    def __clonotypes_data_clean(self, data, chain):
        df = data.copy()
        df = data_proc.filter_clones_data(df, self.tcr_columns_paired[chain], file_dir=self.outputs_path)
        df = data_proc.filter_segments(df, segments_path=self.segments_path, v=self.tcr_columns_paired[chain][1],
                                       j=self.tcr_columns_paired[chain][2], organism=self.species,
                                       file_dir=self.outputs_path)
        return df

    def tcremp_clonotypes(self, chain, unique_clonotypes=False):
        start = time.time()
        data_tt = self.input_data.copy()

        if chain == 'TRA' or chain == 'TRB':
            data_tt = data_tt[~data_tt[self.tcr_columns_paired[chain][0]].isna()].reset_index(drop=True)
            data_tt = self.__clonotypes_data_clean(data_tt, chain)

            data_tt = self.__assign_clones_ids(data_tt, chain)
            if unique_clonotypes:
                data_tt = data_tt.drop_duplicates(self.clonotype_id).reset_index(drop=True)
            data_tt['clone_size'] = data_tt.groupby(self.clonotype_id)[self.input_id].transform('count')
            self.clonotypes[chain] = self.__clonotypes_prep(data_tt, chain)
            self.clonotypes[chain].to_csv(self.clonotypes_path[chain], sep='\t')

            data_tt = data_tt[data_tt[self.clonotype_id].isin(self.clonotypes[chain][self.clonotype_id])]
            self.annot_input[chain] = self.__annot_id(data_tt, self.annotation_id)
        elif chain == 'TRA_TRB':
            data_tt = data_tt[~data_tt[self.tcr_columns_paired['TRA'][0]].isna()].reset_index(drop=True)
            data_tt = data_tt[~data_tt[self.tcr_columns_paired['TRB'][0]].isna()].reset_index(drop=True)
            self.clonotypes[chain] = {}

            for current_chain in ['TRA', 'TRB']:
                data_tt = self.__assign_clones_ids_paired(data_tt, current_chain)
                data_tt = self.__clonotypes_data_clean(data_tt, current_chain)

                self.clonotypes[chain][current_chain] = self.__clonotypes_prep(data_tt, current_chain)
                self.clonotypes[chain][current_chain].to_csv(self.clonotypes_path[chain][current_chain], sep='\t')

            data_tt = self.__assign_clones_ids_paired(data_tt, chain)
            if unique_clonotypes:
                data_tt = data_tt.drop_duplicates(self.clonotype_id).reset_index(drop=True)
            data_tt['clone_size'] = data_tt.groupby(self.clonotype_id)[self.input_id].transform('count')

            for current_chain in ['TRA', 'TRB']:
                data_tt = data_tt[data_tt[self.clonotype_id_dict[current_chain]].isin(
                    self.clonotypes[chain][current_chain][self.clonotype_id])]

            self.annot_input[chain] = self.__annot_id(data_tt.reset_index(drop=True), self.annotation_id)

        else:
            # print('Error. Chain is incorrect. Must be TRA, TRB or TRA_TRB')
            logging.error('Error. Chain is incorrect. Must be TRA, TRB or TRA_TRB')

        end = time.time()
        logging.info(f'Clonotypes extraction time: {end - start}')

    def __data_parse_mirpy(self, chain, olga_human_path, clonotypes_path):
        start = time.time()
        lib = SegmentLibrary.load_default(genes=chain)
        db = Repertoire.load(parser=parser.ClonotypeTableParser(), path=olga_human_path)

        pars = parser.ClonotypeTableParser(lib=lib)

        data_parse = pars.parse(source=clonotypes_path)
        data_parse = [x for x in data_parse if len(x.cdr3aa) in range(self.lower_len_cdr3, self.higher_len_cdr3)]

        end = time.time()
        logging.info(f'parse data for mir: {end - start}')
        return lib, db, data_parse

    def __mir_launch(self, chain, lib, db, data_parse, nproc, chunk_sz):
        aligner = ClonotypeAligner.from_library(lib=lib)
        matcher = DenseMatcher(db, aligner)

        start = time.time()
        res = matcher.match_to_df(data_parse, nproc=nproc)
        end = time.time()
        logging.info(f'Mir launch time: {end - start}')
        return res

    def tcremp_dists_count(self, chain, nproc=None, chunk_sz=100):
        if chain == 'TRA' or chain == 'TRB':
            lib, db, data_parse = self.__data_parse_mirpy(chain, self.prototypes_path[chain],
                                                          self.clonotypes_path[chain])
            res = self.__mir_launch(chain, lib, db, data_parse, nproc, chunk_sz)
            res.to_csv(self.dists_res_path[chain], sep='\t', index=False)
        elif chain == 'TRA_TRB':
            chain_1 = 'TRA'
            lib, db, data_parse = self.__data_parse_mirpy(chain_1, self.prototypes_path[chain_1],
                                                          self.clonotypes_path[chain][chain_1])
            res = self.__mir_launch(chain, lib, db, data_parse, nproc, chunk_sz)
            res.to_csv(self.dists_res_path[chain][chain_1], sep='\t', index=False)
            chain_1 = 'TRB'
            lib, db, data_parse = self.__data_parse_mirpy(chain_1, self.prototypes_path[chain_1],
                                                          self.clonotypes_path[chain][chain_1])
            res = self.__mir_launch(chain, lib, db, data_parse, nproc, chunk_sz)
            res.to_csv(self.dists_res_path[chain][chain_1], sep='\t', index=False)

    def __mir_results_proc(self, chain, res_path_chain, clonotypes_path_chain, clonotype_id_str):
        res_df = pd.read_csv(res_path_chain, sep='\t')
        res_df = res_df.set_axis(['id'] + self.dist_cols_dist[chain], axis=1)
        clonotypes = pd.read_csv(clonotypes_path_chain, sep='\t')
        clonotypes['id'] = clonotypes.index
        res_df = res_df.merge(clonotypes[['id', clonotype_id_str]], on='id').drop('id', axis=1)
        return res_df

    def tcremp_palette(self, labels_list):
        self.palette = ml_utils.make_custom_palette(labels_list)

    def tcremp_dists(self, chain):
        start = time.time()
        if chain == 'TRA' or chain == 'TRB':
            self.dists[chain] = self.__mir_results_proc(chain, self.dists_res_path[chain], self.clonotypes_path[chain],
                                                        self.clonotype_id)
            self.annot[chain] = self.annot_input[chain][self.annot_input[chain][self.clonotype_id].isin(
                list(self.dists[chain][self.clonotype_id]))].reset_index(drop=True)
            self.annot_dists[chain] = self.dists[chain].merge(
                self.annot[chain][[self.clonotype_id, self.annotation_id]]).drop(self.clonotype_id, axis=1,
                                                                                 errors='ignore').sort_values(
                self.annotation_id).reset_index(drop=True)
        elif chain == 'TRA_TRB':
            self.dists[chain] = {}
            for current_chain in ['TRA', 'TRB']:
                self.dists[chain][current_chain] = self.__mir_results_proc(current_chain,
                                                                           self.dists_res_path[chain][current_chain],
                                                                           self.clonotypes_path[chain][current_chain],
                                                                           self.clonotype_id)
                self.annot[chain] = self.annot_input[chain][
                    self.annot_input[chain][self.clonotype_id_dict[chain][current_chain]].isin(
                        list(self.dists[chain][current_chain][self.clonotype_id]))].reset_index(drop=True)

            ## add annotation id to dists
            dists_data = self.annot[chain][
                [self.annotation_id, self.clonotype_id] + list(self.clonotype_id_dict[chain].values())]
            annot_clones = dists_data[[self.annotation_id, self.clonotype_id]]

            dists_data = dists_data.drop(self.annotation_id, axis=1).drop_duplicates().reset_index(drop=True)

            current_chain = 'TRA'
            dists_data_a = dists_data.merge(self.dists[chain][current_chain].rename(
                {self.clonotype_id_dict[current_chain]: self.clonotype_id_dict[chain][current_chain]}, axis=1))
            dists_data_a = dists_data_a.drop(list(self.clonotype_id_dict[chain].values()), axis=1)

            current_chain = 'TRB'
            dists_data_b = dists_data.merge(self.dists[chain][current_chain].rename(
                {self.clonotype_id_dict[current_chain]: self.clonotype_id_dict[chain][current_chain]}, axis=1))
            dists_data_b = dists_data_b.drop(list(self.clonotype_id_dict[chain].values()), axis=1)

            self.dists[chain]['joined'] = dists_data_a.merge(dists_data_b, on=self.clonotype_id)
            self.annot_dists[chain] = self.dists[chain]['joined'].merge(annot_clones).drop(self.clonotype_id,
                                                                                           axis=1)
        else:
            logging.error('Error. Chain is incorrect. Must be TRA, TRB or TRA_TRB')

        # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        end = time.time()
        # self.time_dict[chain]['dist_proc'] = {end - start}
        # print(f'dist_proc: {end - start}')
        logging.info(f'dist_proc: {end - start}')

    def tcremp_pca(self, chain, n_components=None):
        # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        start = time.time()
        if n_components is None:
            n_components = self.__n_components
        if (chain == 'TRA') or (chain == 'TRB'):
            self.pca_clones[chain] = ml_utils.pca_proc(self.dists[chain], self.clonotype_id, n_components)
            self.pca[chain] = self.pca_clones[chain].merge(
                self.annot[chain][[self.clonotype_id, self.annotation_id]]).drop(self.clonotype_id, axis=1,
                                                                                 errors='ignore').sort_values(
                self.annotation_id).reset_index(drop=True)
            self.annot[chain] = self.annot[chain][
                self.annot[chain][self.clonotype_id].isin(list(self.pca_clones[chain][self.clonotype_id]))].reset_index(
                drop=True)

        elif chain == 'TRA_TRB':
            dists_data = self.annot[chain][
                [self.annotation_id, self.clonotype_id] + list(self.clonotype_id_dict[chain].values())]
            annot_clones = dists_data[[self.annotation_id, self.clonotype_id]]

            dists_data = dists_data.drop(self.annotation_id, axis=1).drop_duplicates().reset_index(drop=True)
            for current_chain in ['TRA', 'TRB']:
                self.pca_ad['clones'][current_chain] = ml_utils.pca_proc(self.dists[chain][current_chain],
                                                                         self.clonotype_id,
                                                                         round(n_components)).rename(
                    {self.clonotype_id: self.clonotype_id_dict[chain][current_chain]}, axis=1)
                self.pca_ad['all'][current_chain] = self.pca_ad['clones'][current_chain].merge(
                    self.annot[chain][[self.clonotype_id_dict[chain][current_chain], self.annotation_id]]).drop(
                    self.clonotype_id_dict[chain][current_chain], axis=1, errors='ignore').sort_values(
                    self.annotation_id).reset_index(drop=True)

            self.pca_ad['clones'][chain] = pd.merge(
                dists_data.merge(self.pca_ad['clones']['TRA']).drop(
                    list(self.clonotype_id_dict[chain].values()), axis=1),
                dists_data.merge(self.pca_ad['clones']['TRB']).drop(
                    list(self.clonotype_id_dict[chain].values()), axis=1),
                on=self.clonotype_id)
            self.pca_ad['all'][chain] = self.pca_ad['clones'][chain].merge(annot_clones).drop(self.clonotype_id, axis=1)

            self.pca_clones[chain] = ml_utils.pca_proc(self.dists[chain]['joined'], self.clonotype_id,
                                                       n_components)
            self.pca[chain] = self.pca_clones[chain].merge(annot_clones).drop(self.clonotype_id, axis=1)

            self.annot[chain] = self.annot[chain][
                self.annot[chain][self.clonotype_id].isin(list(self.pca_clones[chain][self.clonotype_id]))].reset_index(
                drop=True)

        end = time.time()
        # self.time_dict[chain]['pca'] = {end - start}
        # print(f'pca: {end - start}')
        logging.info(f'pca: {end - start}')

    def tcremp_tsne(self, chain, ):
        # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        start = time.time()
        self.tsne[chain] = ml_utils.tsne_proc(self.pca[chain], self.annotation_id, self.__tsne_init,
                                              self.__random_state, self.__tsne_perplexity)
        self.tsne_clones[chain] = ml_utils.tsne_proc(self.pca_clones[chain], self.clonotype_id, self.__tsne_init,
                                                     self.__random_state, self.__tsne_perplexity)

        # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        end = time.time()
        # self.time_dict[chain]['tsne'] = {end - start}
        # print(f'tsne: {end - start}')
        logging.info(f'tsne: {end - start}')
