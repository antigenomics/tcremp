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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from kneed import KneeLocator

import tcremp.data_proc as data_proc
import tcremp.ml_utils as ml_utils
import tcremp.motif_logo as motif_logo
import tcremp.metrics as metrics

class TcrempClustering():
    def __init__(self, algo_name='dbscan', threshold=0.7):
        self.annotation_id = 'annotId'
        self.algo_name = algo_name.lower()
        self.threshold = threshold
        self.clstr_labels = {}
        self.binom_res = {}
        self.clstr_metrics = {}
        self.model = {}
        #self.n_clusters = {}
        self.silhouette_n_clusters = {}
        self.silhouette_score = {}
        self.purity = {}
        self.kneedle = {}
        self.poly_degree = 10
        self.coef_dict = {'TRA':0.75, 'TRB': 0.75,'TRA_TRB':0.85}
        self.label = {}
    
    def build_clusters(self, chain, data, label_cl=None, on_pca=True, knee_coef=None):
        self.annotation_id = data.annotation_id
        self.label[chain] = label_cl
        if knee_coef is None:
            knee_coef = self.coef_dict[chain]
        
        annot_clones = data.annot[chain][[data.clonotype_id, self.annotation_id]]
        if on_pca:
            X_data = data.pca_clones[chain].copy()
        else:
            if chain =='TRA_TRB':
                X_data = data.dists[chain]['joined'].copy()
            else:
                X_data = data.dists[chain].copy()
        
        #check_between = False
        if self.algo_name == 'kmeans':
            self.silhouette_clusters(X_data.drop(data.clonotype_id, axis=1), chain)
            algorithm =  KMeans(n_clusters=self.silhouette_n_clusters[chain], random_state=7)
            #check_between = True
        elif self.algo_name == 'dbscan':
            self.knee_coef = knee_coef
            self.eps = {}
            self.eps_by_knn_knee(X_data.drop(data.clonotype_id, axis=1), chain)
            algorithm = DBSCAN(eps=self.eps[chain], min_samples=2)
        else:
            sys.exit(f'Unknown clustering algorithm "{self.algo_name}".')
        
        clstr_labels, self.model[chain] = ml_utils.clstr_model(algorithm, X_data , data.clonotype_id)
        self.clstr_labels[chain] = clstr_labels.merge(annot_clones).drop(data.clonotype_id, axis=1)
        
        if self.label[chain] is not None:
            self.binom_res[chain] = ml_utils.binominal_test(pd.merge(self.clstr_labels[chain],data.annot[chain]), 'cluster', label_cl, self.threshold)
            
            self.binom_res[chain] = self.binom_res[chain].rename({label_cl:'label_cluster'},axis=1)
            
            if -1 in list(self.clstr_labels[chain]['cluster']):
                self.binom_res[chain]['is_cluster'] = self.binom_res[chain].apply(lambda x: x.is_cluster if x.cluster != -1 else 0,axis=1)
                self.binom_res[chain]['label_cluster'] = self.binom_res[chain].apply(lambda x: x.label_cluster if x.cluster != -1 else 'no',axis=1)
        
            #self.clstr_metrics[chain] = ml_utils.clstr_metrics(data.annot[chain][label_cl],self.clstr_labels[chain]['cluster'])
        
            self.clstr_labels[chain] = pd.merge(self.clstr_labels[chain], self.binom_res[chain]
                                                , on='cluster', how='left').sort_values(self.annotation_id).reset_index(
                drop=True)

            self.purity[chain] = ml_utils.count_clstr_purity(self.binom_res[chain])
            #self.silhouette_score[chain] = silhouette_score(X_data, clstr_labels)
            # print(f'purity:{self.purity[chain]}')

        if chain == 'TRA_TRB':
            self.clstr_labels[chain] = pd.merge(self.clstr_labels[chain], data.annot[chain][[self.annotation_id] 
                                                                                   + list(data.tcr_columns_paired['TRA']) 
                                                                                  + list(data.tcr_columns_paired['TRB'])])
        else: 
            self.clstr_labels[chain] = pd.merge(self.clstr_labels[chain],data.annot[chain][[self.annotation_id] + data.tcr_columns_paired[chain]])
    
    
    def clstr_metrics_calc(self, chain, data):
        df = data.annot_input[chain].merge(self.clstr_labels[chain], how='left')
        df['is_cluster'] = df['is_cluster'].fillna(0)
        self.clstr_metrics[chain] = metrics.get_clustermetrics(df, self.label[chain])
        self.clstr_metrics[chain]['total pairs TCR-epitope'] = len(data.annot[chain])
        self.clstr_metrics[chain]['total unique TCRs'] = str(len(data.annot[chain].drop_duplicates(data.clonotype_id)))
        self.clstr_metrics[chain]['total unique epitopes'] = str(len(data.annot[chain].drop_duplicates(self.label[chain])))
        self.clstr_metrics[chain]['label']=self.label[chain]


    def eps_by_knn_knee(self, X_data, chain):
        neighbors=4
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X_data)
        distances, indices = nbrs.kneighbors(X_data)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        
        self.kneedle[chain] = KneeLocator(range(1,len(distances)+1),  #x values
                              distances, # y values
                              S=1.0, #parameter suggested from paper
                              #curve="convex", #parameter from figure
                              curve="concave",
                              interp_method="polynomial",
                              #interp_method="polynomial",        
                              polynomial_degree=self.poly_degree,
                              online = True,
                              direction="increasing", ) #parameter from figure
    
        self.eps[chain] = round(distances[self.kneedle[chain].knee]*self.knee_coef,2)
        if self.eps[chain]==0:
            self.eps[chain] = round(distances.mean()*self.knee_coef,2)
            #self.eps[chain] = round(statistics.median(distances)*self.knee_coef,2)
    
    def plot_knee_normalized(self, chain,
                         title: str = "Normalized Knee Point",
                         xlabel: str =  None,
                         ylabel: str = None,
                        ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.plot(self.kneedle[chain].x_normalized, self.kneedle[chain].y_normalized, "b", label="normalized curve")
        ax.plot(self.kneedle[chain].x_difference, self.kneedle[chain].y_difference, "r", label="difference curve")
        ax.set_xticks(
            np.arange(self.kneedle[chain].x_normalized.min(), self.kneedle[chain].x_normalized.max() + 0.1, 0.1)
        )
        ax.set_yticks(
            np.arange(self.kneedle[chain].y_difference.min(), self.kneedle[chain].y_normalized.max() + 0.1, 0.1)
        )

        ax.vlines(
            self.kneedle[chain].norm_knee,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            linestyles="--",
            label="knee/elbow",
        )
        ax.legend(loc="best")
        
    def plot_knee(self, chain,
                  title: str = "Knee Point",
                  xlabel: str =  None,
                  ylabel: str = None,
                  ax=None): 
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.plot(self.kneedle[chain].x, self.kneedle[chain].y, "b", label="data")
        ax.vlines(
            self.kneedle[chain].knee, ax.get_ylim()[0], ax.get_ylim()[1], linestyles="--", label="knee/elbow"
        )
        ax.legend(loc="best")
    
        
    def silhouette_clusters(self, X_data, chain):
        X_data = X_data.drop(self.annotation_id, axis=1, errors = 'ignore')
    
        data_len = len(X_data)
    
        range_n_clusters = [
            #round(data_len*0.005),
            round(data_len*0.01), round(data_len*0.05) , round(data_len*0.1), round(data_len*0.15)
            ,round(data_len*0.2), round(data_len*0.25), round(data_len*0.3)
            , round(data_len*0.4), round(data_len*0.5), round(data_len*0.6), round(data_len*0.7), round(data_len*0.8), round(data_len*0.9)]
    
        silhouette_avg_scores = ml_utils.silhouette_avg_scores_kmeans(X_data,range_n_clusters)
    
        n_to_try = list(silhouette_avg_scores.keys())[list(silhouette_avg_scores.values()).index(max(silhouette_avg_scores.values()))]
    
        range_n_clusters = [round(n_to_try*0.7), round(n_to_try*0.8), round(n_to_try*0.9), n_to_try, round(n_to_try*1.1) , round(n_to_try*1.2) , round(n_to_try*1.3)]
        silhouette_avg_scores = ml_utils.silhouette_avg_scores_kmeans(X_data,range_n_clusters)

        self.silhouette_n_clusters[chain] = list(silhouette_avg_scores.keys())[list(silhouette_avg_scores.values()).index(max(silhouette_avg_scores.values()))]
        
        
    def __is_cluster_by_between_metric(self, data,chain):
        df = self.clstr_labels[chain].copy()
        df = df[(df['total_cluster']>1)]
        all_cl = list(df['cluster'].drop_duplicates())
    
        df_pca_cl = df[['cluster','annotId']].merge(data.pca[chain])
        df_centrs = pd.DataFrame(self.model[chain].cluster_centers_)
    
        min_between_cl = min(pdist(df_centrs))
        mean_between_cl = statistics.mean(df_centrs)

        all_dists = []
        for i in all_cl:
            df_cl = df_pca_cl[df_pca_cl['cluster']==i].drop(['cluster','annotId'],axis=1)
            centr = df_centrs.iloc[i]
            cl_dists = []
            for _,j in df_cl.iterrows():
                cl_dists.append(cdist(np.array([centr]),np.array([j]),'euclidean')[0][0])
            all_dists.append(statistics.mean(cl_dists))

        cl_mean_dist = statistics.mean(all_dists)
    
        cl_dists = pd.DataFrame([all_cl,all_dists]).T.rename({0:'cluster',1:'dist'},axis=1)

        br = self.binom_res[chain].copy()
        br = br.merge(cl_dists.merge(self.binom_res[chain]),how='left')

        try_dist = statistics.mean([cl_mean_dist,min_between_cl])
        #try_dist = min_between_cl
        #try_dist = statistics.mean([min_between_cl, mean_between_cl])
        br['pred_enriched_min_between']=br.apply(lambda r: 1 if (r.is_cluster == 1) & (r.dist<=try_dist) else 0, axis = 1)

        pred_enriched_between_cls = list(br[br['pred_enriched_min_between']==1]['cluster'])

        self.clstr_labels[chain]['is_cluster_between'] = self.clstr_labels[chain]['cluster'].apply(lambda x: 1 if x in pred_enriched_between_cls else 0)

        
    def __plot_logo(self, clstr_data, chain, c, list_ax, tcr_columns_paired):
        cluster_df = clstr_data[clstr_data['cluster']==c]
        lengs = cluster_df['cdr3aa_len'].drop_duplicates()
        fr_matched = cluster_df['fraction_matched'].drop_duplicates().reset_index(drop=True)[0]
        epi = cluster_df['label_cluster'].drop_duplicates().reset_index(drop=True)[0]
        total_cl = cluster_df['total_cluster'].drop_duplicates().reset_index(drop=True)[0]
        alphabet = [aa for aa in 'ARNDCQEGHILKMFPSTWYVBZX-']
        for l in lengs:
            seqs = cluster_df[cluster_df['cdr3aa_len']==l][tcr_columns_paired[chain][0]].reset_index(drop=True)
            if len(seqs) >= 2:
                motif_logo.plot_amino_logo(seqs, 'title',ax = list_ax[0])
                list_ax[0].set_title(f"{chain}. Cluster: {c} {epi}\nFraction matched:{round(fr_matched,2)}\nCount of cdr3aa: {len(seqs)}")
        plot_v_j = clstr_data[clstr_data['cluster']==c]

        
        plot_v_j = pd.DataFrame(plot_v_j.groupby(tcr_columns_paired[chain][1])[tcr_columns_paired[chain][0]].count().reset_index())
        list_ax[1].pie(plot_v_j[tcr_columns_paired[chain][0]],labels=plot_v_j[tcr_columns_paired[chain][1]])


        plot_v_j = clstr_data[clstr_data['cluster']==c]
        plot_v_j = pd.DataFrame(plot_v_j.groupby(tcr_columns_paired[chain][2])[tcr_columns_paired[chain][0]].count().reset_index())
        list_ax[2].pie(plot_v_j[tcr_columns_paired[chain][0]],labels=plot_v_j[tcr_columns_paired[chain][2]])       
    
    def __plot_logo_paired(self, clstr_data,chain, c, list_ax):
        cluster_df = clstr_data[clstr_data['cluster']==c]
        fr_matched = cluster_df['fraction_matched'].drop_duplicates().reset_index(drop=True)[0]
        epi = cluster_df['label_cluster'].drop_duplicates().reset_index(drop=True)[0]
        total_cl = cluster_df['total_cluster'].drop_duplicates().reset_index(drop=True)[0]
        lengs = cluster_df['a_cdr3aa_len'].drop_duplicates()
        alphabet = [aa for aa in 'ARNDCQEGHILKMFPSTWYVBZX-']
        for l in lengs:
            seqs = cluster_df[cluster_df['a_cdr3aa_len']==l]['a_cdr3aa'].reset_index(drop=True)
            if len(seqs) >= 2:
                motif_logo.plot_amino_logo(seqs, 'title', ax=list_ax[0])
                list_ax[0].set_title(
                    f"{chain}. Cluster: {c} {epi}\nFraction matched:{round(fr_matched, 2)}\nCount of cdr3aa: {len(seqs)}")

        cluster_df = clstr_data[clstr_data['cluster'] == c]
        lengs = cluster_df['b_cdr3aa_len'].drop_duplicates()
        alphabet = [aa for aa in 'ARNDCQEGHILKMFPSTWYVBZX-']
        for l in lengs:
            seqs = cluster_df[cluster_df['b_cdr3aa_len']==l]['b_cdr3aa'].reset_index(drop=True)
            if len(seqs) >= 2:
                #freq = np.zeros((len(alphabet), l))
                #for pos in range(l):
                #    for s in seqs:
                #        freq[alphabet.index(s[pos]), pos] +=1
                #freq_res = freq/freq.sum(axis=0, keepdims=True)
                #motif = pd.DataFrame(freq_res,index=alphabet)
                #motif_logo.plot_amino_logo(motif, 'title',ax = list_ax[1])
                motif_logo.plot_amino_logo(seqs, 'title',ax = list_ax[1])
                list_ax[1].set_title(f"TRB. cdr3aa length: {l}")
    
        
        plot_v_j = clstr_data[clstr_data['cluster']==c]
        plot_v_j = pd.DataFrame(plot_v_j.groupby('TRAV')['a_cdr3aa'].count().reset_index())
        list_ax[2].pie(plot_v_j['a_cdr3aa'],labels=plot_v_j['TRAV'])
        #plot_v_j['count']=plot_v_j.groupby('TRAV').transform('size')
        #plot_v_j = plot_v_j.sort_values('TRAV')
        #sns.histplot(plot_v_j[['count','TRAV']],y='TRAV',ax=list_ax[2])

        plot_v_j = clstr_data[clstr_data['cluster']==c]
        plot_v_j = pd.DataFrame(plot_v_j.groupby('TRAJ')['a_cdr3aa'].count().reset_index())
        list_ax[3].pie(plot_v_j['a_cdr3aa'],labels=plot_v_j['TRAJ'])
        #plot_v_j['count']=plot_v_j.groupby('TRAJ').transform('size')
        #plot_v_j = plot_v_j.sort_values('TRAJ')
        #sns.histplot(plot_v_j[['count','TRAJ']],y='TRAJ',ax=list_ax[3])

        plot_v_j = clstr_data[clstr_data['cluster']==c]
        plot_v_j = pd.DataFrame(plot_v_j.groupby('TRBV')['b_cdr3aa'].count().reset_index())
        list_ax[4].pie(plot_v_j['b_cdr3aa'],labels=plot_v_j['TRBV'])
        #plot_v_j['count']=plot_v_j.groupby('TRBV').transform('size')
        #plot_v_j = plot_v_j.sort_values('TRBV')
        #sns.histplot(plot_v_j[['count','TRBV']],y='TRBV',ax=list_ax[4])

        plot_v_j = clstr_data[clstr_data['cluster']==c]
        plot_v_j = pd.DataFrame(plot_v_j.groupby('TRBJ')['b_cdr3aa'].count().reset_index())
        list_ax[5].pie(plot_v_j['b_cdr3aa'],labels=plot_v_j['TRBJ'])
        #plot_v_j['count']=plot_v_j.groupby('TRBJ').transform('size')
        #plot_v_j = plot_v_j.sort_values('TRBJ')
        #sns.histplot(plot_v_j[['count','TRBJ']],y='TRBJ',ax=list_ax[5])
        
        
    def clstrs_motif(self, data, chain, n_head_clstrs, sfig=None):
        n_cols = 5
        if (chain=='TRA') or (chain=='TRB'):
            br = self.binom_res[chain][self.binom_res[chain]['cluster']!=-1]
            plt_clusters = list(br.sort_values('p_value').head(n_head_clstrs)['cluster'])
            clstr_data = pd.merge(self.clstr_labels[chain],data.annot[chain])
            clstr_data = clstr_data[clstr_data['cluster']!=-1]
        
            clstr_data['cdr3aa_len'] = clstr_data[data.tcr_columns_paired[chain][0]].apply(len)
        
            n_rows = math.ceil(len(plt_clusters)/n_cols)
            
            if sfig is None:
                sfig = plt.figure(figsize=(28,6*n_rows))
                outer_grid = gridspec.GridSpec(n_rows, n_cols,figure=sfig)
            else:
                outer_grid = gridspec.GridSpec(n_rows, n_cols,figure=sfig)


            gs = []
            ax_list = []
            for i in range(len(plt_clusters)):
                cl_row = math.ceil((i+1)/n_cols)-1
                cl_col = i%n_cols
                gs.append(outer_grid[cl_row,cl_col].subgridspec(6, 2))
                ax_list.append([sfig.add_subplot(gs[i][:3, :2])
                                ,sfig.add_subplot(gs[i][4:6, 0])
                                ,sfig.add_subplot(gs[i][4:6, 1])])
    
                self.__plot_logo(clstr_data, chain, plt_clusters[i], ax_list[i],data.tcr_columns_paired)

        elif chain=='TRA_TRB':
            br = self.binom_res[chain][self.binom_res[chain]['cluster']!=-1]
            plt_clusters = list(br.sort_values('p_value').head(n_head_clstrs)['cluster'])
            clstr_data = pd.merge(self.clstr_labels[chain],data.annot[chain])
    
            clstr_data['a_cdr3aa_len'] = clstr_data['a_cdr3aa'].apply(len)
            clstr_data['b_cdr3aa_len'] = clstr_data['b_cdr3aa'].apply(len)
    
            n_rows = math.ceil(len(plt_clusters)/n_cols)

            if sfig is None:
                sfig = plt.figure(figsize=(28,8*n_rows))
                outer_grid = gridspec.GridSpec(n_rows, n_cols,figure=sfig)        
            else:
                outer_grid = gridspec.GridSpec(n_rows, n_cols,figure=sfig)
        
            gs = []
            ax_list = []
            for i in range(len(plt_clusters)):
                cl_row = math.ceil((i+1)/n_cols)-1
                cl_col = i%n_cols
                gs.append(outer_grid[cl_row,cl_col].subgridspec(14, 2))
                ax_list.append([sfig.add_subplot(gs[i][1:4, :2])
                                ,sfig.add_subplot(gs[i][5:8, :2])
                                ,sfig.add_subplot(gs[i][9:11, 0])
                                ,sfig.add_subplot(gs[i][9:11, 1])
                                ,sfig.add_subplot(gs[i][12:14, 0])
                                ,sfig.add_subplot(gs[i][12:14, 1])])
    
                self.__plot_logo_paired(clstr_data, chain, plt_clusters[i], ax_list[i])
        else:
            print('chain is incorrect')