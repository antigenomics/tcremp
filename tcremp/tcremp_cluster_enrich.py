from pathlib import Path
import numpy as np
import pandas as pd
import math

import sys
sys.path.append("../")
import time

import statistics
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import tcremp.data_proc as data_proc
import tcremp.ml_utils as ml_utils
from tcremp.tcremp_pipeline import TcrempPipeline
from tcremp.tcremp_cluster import TcrempClustering


class TcrempPipelineVdjdb(TcrempPipeline):
    def __init__(self,run_name, input_data, clonotype_index = None):
        TcrempPipeline.__init__(self,run_name, input_data, clonotype_index)
        self.clonotypes_pred={}
        #self.vdjdb_path = 'data/VDJdb_pred/vdjdb_data_with_cloneId.txt'
        self.vdjdb_path = { 'TRA' : 'data/VDJdb_pred/vdjdb_data_with_cloneId_TRA.txt', 'TRB' : 'data/VDJdb_pred/vdjdb_data_with_cloneId_TRB.txt',
                               'TRA_TRB': 'data/VDJdb_pred/vdjdb_data_with_cloneId_TRA_TRB.txt'}
        self.vdjdb_clonotypes_path = { 'TRA' : 'data/VDJdb_pred/clonotypes_TRA.txt', 'TRB' : 'data/VDJdb_pred/clonotypes_TRB.txt',
                               'TRA_TRB': {'TRA' : 'data/VDJdb_pred/clonotypes_paired_TRA.txt', 'TRB' : 'data/VDJdb_pred/clonotypes_paired_TRB.txt'}}
        self.vdjdb_dists_res_path = {'TRA' : 'data/VDJdb_pred/res_TRA.txt', 'TRB': 'data/VDJdb_pred/res_TRB.txt',
                              'TRA_TRB':{'TRA' : 'data/VDJdb_pred/res_paired_TRA.txt', 'TRB': 'data/VDJdb_pred/res_paired_TRB.txt'}}
        self.vdjdb = {}
        self.vdjdb_clonotype_id = 'vdjdb_cloneId'
        self.vdjdb_clonotype_id_dict = {'TRA': 'vdjdb_cloneId','TRB': 'vdjdb_cloneId','TRA_TRB': {'TRA':'vdjdb_cloneId_TRA', 'TRB':'vdjdb_cloneId_TRB'}}
        self.data_type = 'data_type'
        self.label = 'antigen.epitope_freq'
        #self.label = 'antigen.epitope'
        
        self.dists_dubs = {}
        
        
    def __assign_clone_ids_with_vdjdb_paired(self, data, chain):
        df = data.copy()
        if (chain=='TRA') or (chain=='TRB'):
            #df[self.clonotype_id_dict['TRA_TRB'][chain]]=df.groupby(self.tcr_columns_paired[chain],dropna=False).ngroup()
            vdjdb_df = self.vdjdb['TRA_TRB'].copy()
            vdjdb_df = vdjdb_df[self.tcr_columns_paired[chain] + [self.vdjdb_clonotype_id_dict['TRA_TRB'][chain]]].drop_duplicates().reset_index(drop=True)
            
            df = df.merge(df.merge(vdjdb_df)[[self.input_id,self.vdjdb_clonotype_id_dict['TRA_TRB'][chain]]], how='left').reset_index(drop=True)
            t = df[df[self.vdjdb_clonotype_id_dict['TRA_TRB'][chain]].isna()]
            t = self._TcrempPipeline__assign_clones_ids_paired(t, chain)
            t[self.clonotype_id_dict['TRA_TRB'][chain]] = t[self.clonotype_id_dict['TRA_TRB'][chain]] + (max(self.vdjdb['TRA_TRB'][self.vdjdb_clonotype_id_dict['TRA_TRB'][chain]] +1))
        
            df = df.merge(t[[self.input_id,self.clonotype_id_dict['TRA_TRB'][chain]]],how='left')
            df[self.clonotype_id_dict['TRA_TRB'][chain]] = df.apply(lambda x: x[self.vdjdb_clonotype_id_dict['TRA_TRB'][chain]] if math.isnan(x[self.clonotype_id_dict['TRA_TRB'][chain]])  else x[self.clonotype_id_dict['TRA_TRB'][chain]],axis=1)
        
        if chain=='TRA_TRB':
            #df[self.clonotype_id]=df.groupby(self.tcr_columns_paired['TRA']+self.tcr_columns_paired['TRB'],dropna=False).ngroup()
            
            df = df.merge(df.merge(self.vdjdb[chain][self.tcr_columns_paired['TRA'] + self.tcr_columns_paired['TRB'] + [self.vdjdb_clonotype_id]].drop_duplicates())[[self.input_id,self.vdjdb_clonotype_id]], how='left').reset_index(drop=True)
            t = df[df[self.vdjdb_clonotype_id].isna()]
            t = self._TcrempPipeline__assign_clones_ids_paired(t, chain)
            t[self.clonotype_id] = t[self.clonotype_id] + (max(self.vdjdb[chain][self.vdjdb_clonotype_id] +1))
        
            df = df.merge(t[[self.input_id,self.clonotype_id]],how='left')
            df[self.clonotype_id] = df.apply(lambda x: x[self.vdjdb_clonotype_id] if math.isnan(x[self.clonotype_id]) else x[self.clonotype_id],axis=1)
        return df
    
    
    def __assign_clone_ids_with_vdjdb(self, data,chain):
        df = data.copy()
        df = df.merge(df.merge(self.vdjdb[chain][self.tcr_columns_paired[chain] + [self.vdjdb_clonotype_id]].drop_duplicates())[[self.input_id,self.vdjdb_clonotype_id]], how='left').reset_index(drop=True)
        t = df[df[self.vdjdb_clonotype_id].isna()]
        t = self._TcrempPipeline__assign_clones_ids(t, chain)
        t[self.clonotype_id] = t[self.clonotype_id] + (max(self.vdjdb[chain][self.vdjdb_clonotype_id] +1))
        
        df = df.merge(t[[self.input_id,self.clonotype_id]],how='left')
        df[self.clonotype_id] = df.apply(lambda x: x[self.vdjdb_clonotype_id] if math.isnan(x[self.clonotype_id])  else x[self.clonotype_id],axis=1)
        return df
    
    
    def tcremp_clonotypes(self,chain, unique_clonotypes=False):
        self.time_dict[chain]={}
        df = self.input_data.copy()
        self.vdjdb[chain] = pd.read_csv(self.vdjdb_path[chain],sep='\t')
        #self.vdjdb[chain] = self.vdjdb[chain][self.vdjdb[chain]['antigen.epitope_freq']!='other']
        if (chain=='TRA') or (chain=='TRB'):
            df = df[~df[self.tcr_columns_paired[chain][0]].isna()].reset_index(drop=True)
            df = self._TcrempPipeline__clonotypes_data_clean(df, chain) #050624
            
            df = self.__assign_clone_ids_with_vdjdb(df, chain)
            if unique_clonotypes:
                df = df.drop_duplicates(self.clonotype_id).reset_index(drop=True)
            df['clone_size'] = df.groupby(self.clonotype_id)[self.input_id].transform('count')
            
            #050624data_chain_1 = df.rename(self._TCRemP__rename_tcr_columns_paired[chain],axis=1)
            #050624data_chain_1['chain']=chain
            
            #050624self.clonotypes_pred[chain] = self._TCRemP__clonotypes_prep(data_chain_1,  chain, self.tcr_columns, self.clonotype_id)
            self.clonotypes_pred[chain] = self._TCRemP__clonotypes_prep(df, chain) #050624
            self.clonotypes_pred[chain].to_csv(self.clonotypes_path[chain], sep='\t')
            
            vdjdb = self.vdjdb[chain].copy()
            vdjdb[self.clonotype_id] = vdjdb[self.vdjdb_clonotype_id]
            vdjdb = vdjdb[~vdjdb[self.tcr_columns_paired[chain][0]].isna()].reset_index(drop=True)
            vdjdb[self.data_type]='train'
            #df[self.label]='no'
            df[self.data_type]='pred'
            df = pd.concat([vdjdb, df])
            #050624df_cl = df.rename(self._TCRemP__rename_tcr_columns_paired[chain],axis=1)
            #050624df_cl['chain']=chain
            
            #050624self.clonotypes[chain] = self._TCRemP__clonotypes_prep(df_cl, chain, self.tcr_columns, self.clonotype_id)
            self.clonotypes[chain] = self._TcrempPipeline__clonotypes_prep(df, chain) #050624
            
            df = df[df[self.clonotype_id].isin(self.clonotypes[chain][self.clonotype_id])]
            self.annot_input[chain] = self._TcrempPipeline__annot_id(df.reset_index(drop=True), self.annotation_id)

            
        elif chain=='TRA_TRB':
            df = df[~df[self.tcr_columns_paired['TRA'][0]].isna()].reset_index(drop=True)
            df = df[~df[self.tcr_columns_paired['TRB'][0]].isna()].reset_index(drop=True)
            vdjdb = self.vdjdb[chain].copy()
            #vdjdb[self.clonotype_id] = vdjdb[self.vdjdb_clonotype_id]
            vdjdb[self.clonotype_id_dict[chain]['TRA']] = vdjdb[self.vdjdb_clonotype_id_dict[chain]['TRA']]
            vdjdb[self.clonotype_id_dict[chain]['TRB']] = vdjdb[self.vdjdb_clonotype_id_dict[chain]['TRB']]
            vdjdb = vdjdb[~vdjdb[self.tcr_columns_paired['TRA'][0]].isna()].reset_index(drop=True)
            vdjdb = vdjdb[~vdjdb[self.tcr_columns_paired['TRB'][0]].isna()].reset_index(drop=True)
            
            self.clonotypes[chain] = {}
            self.clonotypes_pred[chain] = {}
            
            chain_1 = 'TRA'
            df = self.__assign_clone_ids_with_vdjdb_paired(df, chain_1)
            df = self._TcrempPipeline__clonotypes_data_clean(df, chain_1) #050624
            #050624data_chain_1 = df.copy()
            #050624data_chain_1 = data_chain_1.rename(self._TCRemP__rename_tcr_columns_paired[chain_1],axis=1)
            #050624data_chain_1['chain']=chain_1
            #050624self.clonotypes_pred[chain][chain_1] = self._TCRemP__clonotypes_prep(data_chain_1, chain_1, self.tcr_columns, self.clonotype_id)
            self.clonotypes_pred[chain][chain_1] = self._TcrempPipeline__clonotypes_prep(df, chain_1) #050624
            self.clonotypes_pred[chain][chain_1].to_csv(self.clonotypes_path[chain][chain_1], sep='\t')
            
            #050624vdjdb_1 = vdjdb.copy()
            #050624vdjdb_1 = vdjdb_1.rename(self._TCRemP__rename_tcr_columns_paired[chain_1],axis=1)
            #050624vdjdb_1['chain']=chain_1
            #050624print(vdjdb_1.head(1))
            #050624data_chain_1 = pd.concat([vdjdb_1, data_chain_1])
            data_chain_1 = pd.concat([vdjdb, df]) #050624
            
            #050624self.clonotypes[chain][chain_1] = self._TCRemP__clonotypes_prep(data_chain_1, chain_1, self.tcr_columns, self.clonotype_id)
            self.clonotypes[chain][chain_1] = self._TcrempPipeline__clonotypes_prep(data_chain_1, chain_1) #050624
            
            
            chain_1 = 'TRB'
            df = self.__assign_clone_ids_with_vdjdb_paired(df, chain_1)
            df = self._TcrempPipeline__clonotypes_data_clean(df, chain_1) #050624
            #050624data_chain_1 = df.copy()
            #050624data_chain_1 = data_chain_1.rename(self._TCRemP__rename_tcr_columns_paired[chain_1],axis=1)
            #050624data_chain_1['chain']=chain_1
            #050624self.clonotypes_pred[chain][chain_1] = self._TCRemP__clonotypes_prep(data_chain_1, chain_1, self.tcr_columns, self.clonotype_id)
            self.clonotypes_pred[chain][chain_1] = self._TcrempPipeline__clonotypes_prep(df, chain_1) #050624
            self.clonotypes_pred[chain][chain_1].to_csv(self.clonotypes_path[chain][chain_1], sep='\t')
            
            #050624vdjdb_1 = vdjdb.copy()
            #050624vdjdb_1 = vdjdb_1.rename(self._TCRemP__rename_tcr_columns_paired[chain_1],axis=1)
            #050624vdjdb_1['chain']=chain_1
            #050624data_chain_1 = pd.concat([vdjdb_1, data_chain_1])
            data_chain_1 = pd.concat([vdjdb, df]) #050624
            
            #050624self.clonotypes[chain][chain_1] = self._TCRemP__clonotypes_prep(data_chain_1, chain_1, self.tcr_columns, self.clonotype_id)
            self.clonotypes[chain][chain_1] = self._TcrempPipeline__clonotypes_prep(data_chain_1, chain_1) #050624
            
            
            df = self.__assign_clone_ids_with_vdjdb_paired(df, chain)
            if unique_clonotypes:
                df = df.drop_duplicates(self.clonotype_id).reset_index(drop=True)
            
            vdjdb[self.clonotype_id] = vdjdb[self.vdjdb_clonotype_id]
            
            vdjdb[self.data_type]='train'
            #df[self.label]='no'
            df[self.data_type]='pred'
            df = pd.concat([vdjdb, df])
            
            df['clone_size'] = df.groupby(self.clonotype_id)[self.input_id].transform('count')
            
            chain_1 = 'TRA'
            df = df[df[self.clonotype_id_dict[chain_1]].isin(self.clonotypes[chain][chain_1][self.clonotype_id])]
            chain_1 = 'TRB'
            df = df[df[self.clonotype_id_dict[chain_1]].isin(self.clonotypes[chain][chain_1][self.clonotype_id])]
            
            self.annot_input[chain] = self._TcrempPipeline__annot_id(df.reset_index(drop=True), self.annotation_id)

        else:
            print('Error. Chain is incorrect. Must be TRA, TRB or TRA_TRB')
            
    
    def tcremp_dists(self, chain):
        
        if (chain=='TRA') or (chain=='TRB'):
            dists_vdjdb = self._TcrempPipeline__mir_results_proc(chain, self.vdjdb_dists_res_path[chain], self.vdjdb_clonotypes_path[chain], self.clonotype_id)
        
            dists_pred = self._TcrempPipeline__mir_results_proc(chain, self.dists_res_path[chain], self.clonotypes_path[chain], self.clonotype_id)
            self.dists_dubs[chain] = pd.concat([dists_vdjdb,dists_pred])
            self.dists[chain] = self.dists_dubs[chain].drop_duplicates(self.clonotype_id).reset_index(drop=True)
            
            self.annot[chain] = self.annot_input[chain][self.annot_input[chain][self.clonotype_id].isin(list(self.dists[chain][self.clonotype_id]))].reset_index(drop=True) ##!
            
            self.annot_dists[chain] = self.dists[chain].merge(self.annot[chain][[self.clonotype_id,self.annotation_id]]).drop(self.clonotype_id, axis=1, errors = 'ignore').sort_values(self.annotation_id).reset_index(drop=True) ##230524 !
            
            #self.annot[chain] = self.annot_input[chain][self.annot_input[chain][self.clonotype_id].isin(list(self.dists[chain][self.clonotype_id]))].reset_index(drop=True)
        elif chain=='TRA_TRB':
            self.dists_dubs[chain] = {}
            self.dists[chain] = {}
            for chain_1 in ['TRA', 'TRB']:
                dists_vdjdb = self.__mir_results_proc(chain_1, self.vdjdb_dists_res_path[chain][chain_1],
                                                                     self.vdjdb_clonotypes_path[chain][chain_1],
                                                                     self.clonotype_id)
                dists_pred = self.__mir_results_proc(chain_1, self.dists_res_path[chain][chain_1],
                                                                    self.clonotypes_path[chain][chain_1],
                                                                    self.clonotype_id)


            chain_1 = 'TRB'
            dists_vdjdb = self._TcrempPipeline__mir_results_proc(chain_1, self.vdjdb_dists_res_path[chain][chain_1], self.vdjdb_clonotypes_path[chain][chain_1], self.clonotype_id)
            dists_pred = self._TcrempPipeline__mir_results_proc(chain_1, self.dists_res_path[chain][chain_1], self.clonotypes_path[chain][chain_1], self.clonotype_id)
            
            self.dists_dubs[chain][chain_1] = pd.concat([dists_vdjdb,dists_pred])
            self.dists[chain][chain_1] = self.dists_dubs[chain][chain_1].drop_duplicates(self.clonotype_id).reset_index(drop=True)  
            self.annot[chain] = self.annot_input[chain][self.annot_input[chain][self.clonotype_id_dict[chain][chain_1]].isin(list(self.dists[chain][chain_1][self.clonotype_id]))].reset_index(drop=True)
            
            ## add annotation id to dists
            dists_data = self.annot[chain][[self.annotation_id, self.clonotype_id] +list(self.clonotype_id_dict[chain].values())] ##230524
            annot_clones = dists_data[[self.annotation_id, self.clonotype_id]] ##230524            
            
            dists_data = dists_data.drop(self.annotation_id,axis=1).drop_duplicates().reset_index(drop=True) ##230524
            
            chain_1='TRA' ##230524
            dists_data_a = dists_data.merge(self.dists[chain][chain_1].rename({self.clonotype_id_dict[chain_1]:self.clonotype_id_dict[chain][chain_1]},axis=1)) ##230524
            dists_data_a = dists_data_a.drop(list(self.clonotype_id_dict[chain].values()),axis=1) ##230524
            
            chain_1='TRB'
            dists_data_b = dists_data.merge(self.dists[chain][chain_1].rename({self.clonotype_id_dict[chain_1]:self.clonotype_id_dict[chain][chain_1]},axis=1)) ##230524
            dists_data_b = dists_data_b.drop(list(self.clonotype_id_dict[chain].values()),axis=1) ##230524
            
            self.dists[chain]['joined'] = dists_data_a.merge(dists_data_b, on = self.clonotype_id) ##230524
            self.annot_dists[chain] = self.dists[chain]['joined'].merge(annot_clones).drop(self.clonotype_id,axis=1) ##230524
            
        else:
            print('Error. Chain is incorrect. Must be TRA, TRB or TRA_TRB') 
    

class TcrempPipelinePred(TcrempClustering):
    def __init__(self, algo_name='dbscan', threshold=0.7):
        TcrempClustering.__init__(self, algo_name, threshold)
        self.annotation_id = 'annotId'
        self.data_type = 'data_type'
        #self.n_clusters = {}
        self.binom_res_train = {}
        self.clstr_metrics_train = {}
    
    def clstr_pred(self, chain, data, label_cl, model='dbscan', on_pca=True):
        self.build_clusters(chain, data, label_cl, on_pca)
        
        y_data_train = data.annot[chain][data.annot[chain][self.data_type]=='train'][label_cl]
                
        clstrs_labels_data_annot_train = pd.merge(self.clstr_labels[chain], data.annot[chain][data.annot[chain][self.data_type]=='train'])
        
        self.binom_res_train[chain] = ml_utils.binominal_test(clstrs_labels_data_annot_train, 'cluster', label_cl, self.threshold)
        
        #self.binom_res_train[chain]['is_cluster_train']= self.binom_res_train[chain]['total_cluster'].apply(lambda x: 1 if x>1 else 0)
        #self.binom_res_train[chain]['enriched_clstr_train'] =self.binom_res_train[chain].apply(lambda x:1 
        #                                                                     if (x.fraction_matched>=self.threshold)
        #                                                                     and (x.is_cluster==1) else 0,axis=1)
        self.clstr_metrics[chain] = ml_utils.clstr_metrics(clstrs_labels_data_annot_train[label_cl], clstrs_labels_data_annot_train['cluster'])
        
        self.clstr_labels[chain] = pd.merge(self.clstr_labels[chain], self.binom_res_train[chain].rename({label_cl:'label_cluster_train'
                                                                                                          ,'fraction_matched':'fraction_matched_train'
                                                                                                          ,'p_value':'p_value_train'
                                                                                                          ,'is_cluster':'is_cluster_train'
                                                                                                         ,'enriched_clstr':'enriched_clstr_train'},axis=1)[['cluster',
                                                                                                                                              'label_cluster_train',
                                                                                                                                              'fraction_matched_train',
                                                                                                                                              'p_value_train',
                                                                                                                                              'is_cluster_train',
                                                                                                                                              'enriched_clstr_train']]
                                            , on='cluster',how='left').sort_values(self.annotation_id).reset_index(drop=True)        
        if chain == 'TRA_TRB':
            self.clstr_labels[chain] = self.clstr_labels[chain].merge(data.annot[chain][[self.annotation_id, self.data_type] 
                                                                                   + list(data.tcr_columns_paired['TRA']) 
                                                                                  + list(data.tcr_columns_paired['TRB'])])
        else: 
            self.clstr_labels[chain] = self.clstr_labels[chain].merge(data.annot[chain][[self.annotation_id, self.data_type] + data.tcr_columns_paired[chain]])
            
        self.train_purity = ml_utils.count_clstr_purity(self.binom_res_train[chain].rename({'is_cluster_train':'is_cluster'},axis=1))
        #self.train_mean_fraction_matched = statistics.mean(self.binom_res_train[chain][self.binom_res_train[chain]['is_cluster_train']==1]['fraction_matched'])
        #self.train_median_fraction_matched = statistics.median(self.binom_res_train[chain][self.binom_res_train[chain]['is_cluster_train']==1]['fraction_matched'])
        #print(f'train mean fraction_matched only clusters: {self.train_mean_fraction_matched}')
        #print(f'train median fraction_matched only clusters: {self.train_median_fraction_matched}')
        print(f'train purity:{self.train_purity}')    

class TcrempPipelinePred2(TcrempClustering):
    def __init__(self, algo_name='dbscan', threshold=0.7):
        TcrempClustering.__init__(self, algo_name, threshold)
        self.annotation_id = 'annotId'
        self.data_type = 'data_type'
        #self.n_clusters = {}
        self.binom_res_train = {}
        self.clstr_metrics_train = {}
    
    def clstr_pred(self, chain, data, label_cl, model=None):
        
        self.build_clusters(chain, data, label_cl, model)
        
        y_data_train = data.annot[chain][data.annot[chain][self.data_type]=='train'][label_cl]
                
        clstrs_labels_data_annot_train = pd.merge(self.clstr_labels[chain], data.annot[chain][data.annot[chain][self.data_type]=='train'])
        
        self.binom_res_train[chain] = ml_utils.binominal_test(clstrs_labels_data_annot_train, 'cluster', label_cl, self.threshold)
        
        #self.binom_res_train[chain]['is_cluster_train']= self.binom_res_train[chain]['total_cluster'].apply(lambda x: 1 if x>1 else 0)
        #self.binom_res_train[chain]['enriched_clstr_train'] =self.binom_res_train[chain].apply(lambda x:1 
        #                                                                     if (x.fraction_matched>=self.threshold)
        #                                                                     and (x.is_cluster==1) else 0,axis=1)
        self.clstr_metrics[chain] = ml_utils.clstr_metrics(clstrs_labels_data_annot_train[label_cl], clstrs_labels_data_annot_train['cluster'])
        
        self.clstr_labels[chain] = pd.merge(self.clstr_labels[chain], self.binom_res_train[chain].rename({label_cl:'label_cluster_train'
                                                                                                          ,'fraction_matched':'fraction_matched_train'
                                                                                                          ,'p_value':'p_value_train'
                                                                                                          ,'is_cluster':'is_cluster_train'
                                                                                                         ,'enriched_clstr':'enriched_clstr_train'},axis=1)[['cluster',
                                                                                                                                              'label_cluster_train',
                                                                                                                                              'fraction_matched_train',
                                                                                                                                              'p_value_train',
                                                                                                                                              'is_cluster_train',
                                                                                                                                              'enriched_clstr_train']]
                                            , on='cluster',how='left').sort_values(self.annotation_id).reset_index(drop=True)
        
        if chain == 'TRA_TRB':
            self.clstr_labels[chain] = self.clstr_labels[chain].merge(data.annot[chain][[self.annotation_id, self.data_type] 
                                                                                   + list(self.tcr_columns_paired['TRA'].values()) 
                                                                                  + list(self.tcr_columns_paired['TRB'].values())])
        else: 
            self.clstr_labels[chain] = self.clstr_labels[chain].merge(data.annot[chain][[self.annotation_id, self.data_type] + data.tcr_columns])
        
        self.train_purity = ml_utils.count_clstr_purity(self.binom_res_train[chain].rename({'is_cluster_train':'is_cluster'},axis=1))
        #self.train_mean_fraction_matched = statistics.mean(self.binom_res_train[chain][self.binom_res_train[chain]['is_cluster_train']==1]['fraction_matched'])
        #self.train_median_fraction_matched = statistics.median(self.binom_res_train[chain][self.binom_res_train[chain]['is_cluster_train']==1]['fraction_matched'])
        #print(f'train mean fraction_matched only clusters: {self.train_mean_fraction_matched}')
        #print(f'train median fraction_matched only clusters: {self.train_median_fraction_matched}')
        print(f'train purity:{self.train_purity}')    
        