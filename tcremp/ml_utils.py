import logging

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay

from sklearn.model_selection import cross_validate

from sklearn.pipeline import make_pipeline
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score, silhouette_score

from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import warnings

warnings.filterwarnings("ignore")

colors = ['red', 'cyan', 'lime', 'darkgreen', 'gold', 'pink', 'lightsalmon', 'yellow', 'maroon', 'blue', 'teal',
          'orange', 'olive', 'indigo', 'fuchsia', 'palegreen', 'crimson', 'navy', 'black']


def pca_proc(res_df, id_column='id', n_components=100, plot=False):
    data_proc = res_df.drop(id_column, axis=1, errors='ignore')
    logging.info(data_proc)
    pca = PCA(n_components=min([n_components, data_proc.shape[0], data_proc.shape[1]]))
    pca.fit(StandardScaler().fit_transform(data_proc))

    if plot:
        plt.plot(pca.explained_variance_ratio_, 'bx')
        plt.xscale('log')
        plt.ylabel('Explained Variance')
        plt.xlabel('Components')
        plt.xlim(.9, pca.n_components_)

    pca_data = pd.DataFrame(pca.transform(data_proc))

    pca_data = pca_data.rename(columns=lambda x: f'PC{x}')

    pca_data[id_column] = res_df[id_column]

    return pca_data


def tsne_proc(proc_df, id_column='id', init='pca', random_state=7, perplexity=15):
    X_embedded = TSNE(n_components=2, init=init,
                      random_state=random_state, perplexity=perplexity).fit_transform(
        proc_df.drop(id_column, axis=1, errors='ignore'))
    tsne_data = pd.DataFrame(data=X_embedded, columns=['DM1', 'DM2'])

    tsne_data[id_column] = proc_df[id_column]
    return tsne_data


def pc_anova(data, pc_n, group):
    pc_anova = pd.DataFrame(columns=['pc', 'F', 'pvalue'])

    for n in range(0, pc_n):
        F, pval = f_classif(np.array(data[n]).reshape(-1, 1), data[group])
        pc_anova = pd.concat([pc_anova, pd.DataFrame(data={'pc': [n], 'F': [float(F)], 'pvalue': [float(pval)]})])
    pc_anova = pc_anova.sort_values('pvalue')
    return pc_anova


def make_custom_palette(labels_list):
    color_n = len(labels_list)
    # palette = sns.color_palette("bright", color_n).as_hex()
    palette = colors[0:color_n]
    custom_palette = {labels_list[i]: palette[i] for i in range(color_n)}
    custom_palette['other'] = 'lightgrey'
    return custom_palette


def tsne_plot(data_plot, to_color, title, to_size=None, legend=True, custom_palette=None, ax=None, ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    if to_size is None:
        sns.scatterplot(x='DM1', y='DM2', data=data_plot.sort_values(to_color), hue=to_color, s=2, legend=legend,
                        palette=custom_palette, ax=ax)
    else:
        sns.scatterplot(x='DM1', y='DM2', data=data_plot.sort_values(to_color), hue=to_color, size=to_size,
                        sizes=(2, 100), legend=legend, palette=custom_palette, ax=ax)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# Metrics
# to check fraction_matched_exp
def binominal_test(df, cluster, group, threshold=0.7):
    binom_df = df.copy()
    binom_df['total_cluster'] = binom_df.groupby(cluster)[cluster].transform('count')
    binom_df['total_group'] = binom_df.groupby(group)[group].transform('count')
    binom_df['count_matched'] = binom_df.groupby([group, cluster])[group].transform('count')
    binom_df['fraction_matched'] = binom_df['count_matched'] / binom_df['total_cluster']
    binom_df['fraction_matched_exp'] = binom_df['total_group'] / len(binom_df.index)
    binom_df['p_value'] = binom_df.apply(
        lambda row: stats.binomtest(row['count_matched'], n=row['total_cluster'], p=row['fraction_matched_exp'],
                                    alternative='greater').pvalue, axis=1)
    binom_df_cluster = binom_df[
        [group, cluster, 'total_cluster', 'total_group', 'count_matched', 'fraction_matched', 'fraction_matched_exp',
         'p_value']].drop_duplicates().sort_values('p_value')
    binom_df_cluster['is_cluster'] = binom_df_cluster.apply(
        lambda x: 1 if (x.total_cluster > 1) and (x.cluster != -1) else 0, axis=1)
    binom_df_cluster['enriched_clstr'] = binom_df_cluster.apply(lambda x: 1
    if (x.fraction_matched >= threshold)
       and (x.is_cluster == 1) else 0, axis=1)
    binom_df_cluster = binom_df_cluster.sort_values(['fraction_matched'], ascending=False)
    binom_df_cluster = binom_df_cluster.drop_duplicates('cluster', keep='first')
    return binom_df_cluster


def clsf_metrics(y_test, pred, average='micro'):
    return {
        'f1': f1_score(y_test, pred, average=average),
        'precision': precision_score(y_test, pred, average=average),
        'recall': recall_score(y_test, pred, average=average),
    }


def count_clstr_purity(binom_res):
    binom_res_clstr = binom_res[binom_res['is_cluster'] == 1]
    if len(binom_res_clstr) != 0:
        return sum(binom_res_clstr['count_matched']) / sum(binom_res_clstr['total_cluster'])


def clstr_metrics(y_data, labels):
    return {
        'homogeneity_score': homogeneity_score(y_data, labels),
        'completeness_score': completeness_score(y_data, labels),
        'v_measure_score': v_measure_score(y_data, labels),
        'adjusted_rand_score': adjusted_rand_score(y_data, labels),
        'adjusted_mutual_info_score': adjusted_mutual_info_score(y_data, labels),
    }


## Clustering
def clstr_model(model, proc_df, id_column):
    model.fit(proc_df.drop(id_column, axis=1, errors='ignore'))
    clustering_data = pd.DataFrame(model.labels_, columns=['cluster'])

    clustering_data[id_column] = proc_df[id_column]

    return clustering_data, model


def kmeans_proc(proc_df, id_column, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
        proc_df.drop(id_column, axis=1, errors='ignore'))
    kmeans_data = pd.DataFrame(kmeans.labels_, columns=['cluster'])

    kmeans_data[id_column] = proc_df[id_column]

    return kmeans_data


##Classification
def generate_negative_pairs(positive_pairs, n, p1_col, p2_col):
    negative_pairs = []
    i = 0
    while i < n:
        p1 = positive_pairs.sample(n=1).reset_index(drop=True)
        p2 = positive_pairs.sample(n=1).reset_index(drop=True)
        pair = {p1_col: p1[p1_col][0], p2_col: p2[p2_col][0]}
        if (len(positive_pairs[
                    (positive_pairs[p1_col] == pair[p1_col]) & (positive_pairs[p2_col] == pair[p2_col])]) == 0) and (
                pair not in negative_pairs):
            negative_pairs.append(pair)
            i += 1
    return pd.DataFrame(negative_pairs)


def roc_auc_count_binary(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    return auc(fpr, tpr)


def roc_auc_count(y_test_curv, y_pred_curv):
    n_classes = y_test_curv.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_curv[:, i], y_pred_curv[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc


def plot_roccurve_multi(classes_list, y_test_curv, y_score, title, ax, custom_palette=None, test_acc=None,
                        f1_weighted=None, show_legend=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    for class_id in range(len(classes_list)):
        if custom_palette is None:
            color = None  # colors[class_id]
        else:
            color = custom_palette[classes_list[class_id]]
        if sum(y_test_curv[:, class_id]) != 0:
            RocCurveDisplay.from_predictions(
                y_test_curv[:, class_id],
                y_score[:, class_id],
                name=f"ROC curve for {classes_list[class_id]}",
                ax=ax,
                color=color,
                plot_chance_level=(class_id == 2), )
    if f1_weighted is not None:
        print(
            f"accuracy: {round(test_acc, 2)}\nf1_weighted:{round(f1_weighted['f1'], 2)}\nprecision:{round(f1_weighted['precision'], 2)}\nrecall:{round(f1_weighted['recall'], 2)}")
        # ax.text(1.7,0.2,f"accuracy: {round(test_acc,2)}\nf1_weighted:{round(f1_weighted['f1'],2)}\nprecision:{round(f1_weighted['precision'],2)}\nrecall:{round(f1_weighted['recall'],2)}")
    ax.set_title(title)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    else:
        ax.get_legend().remove()


##Evaluation
def silhouette_avg_scores_kmeans(X, range_n_clusters):
    silhouette_avgs = {}
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        silhouette_avgs[n_clusters] = silhouette_avg
    return silhouette_avgs


def silhouette_avg_scores_model(X, range_n_clusters, model_name):
    silhouette_avgs = {}
    for n_clusters in range_n_clusters:
        # clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        clusterer = init_clstr_model(model_name, pd.Series(data=[n_clusters], index=['n_clusters']))
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        silhouette_avgs[n_clusters] = silhouette_avg
    return silhouette_avgs


def silhouette_avg_scores_kmeans_with_plot(X, range_n_clusters):
    silhouette_avgs = {}
    for n_clusters in range_n_clusters:

        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        silhouette_avgs[n_clusters] = silhouette_avg

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters. N=" + str(n_clusters))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    return silhouette_avgs


def clf_cross_validate(clfs, X_train, y_train):
    for classifier in clfs:
        steps = [('model', classifier)]
        pipeline = Pipeline(steps=steps)
        # pipeline.set_params(clf = classifier)
        scores = cross_validate(pipeline, X_train, y_train)
        print('-----------------------------------')
        print(str(classifier))
        print('-----------------------------------')
        for key, values in scores.items():
            print(key, ' mean ', values.mean())
            print(key, ' std ', values.std())


def connectivity_kneighbors_graph(X, n_neighbors):
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    return connectivity


def init_clstr_model(model_name, r):
    if model_name == 'KMeans':
        from sklearn.cluster import KMeans
        clstr = KMeans(n_clusters=r.n_clusters, random_state=12)
    elif model_name == 'Birch':
        from sklearn.cluster import Birch
        clstr = Birch(n_clusters=r.n_clusters)
    elif model_name == 'DBSCAN':
        from sklearn.cluster import DBSCAN
        clstr = DBSCAN(eps=r.eps)
    elif model_name == 'Agglomerativ':
        from sklearn.cluster import AgglomerativeClustering
        clstr = AgglomerativeClustering(n_clusters=r.n_clusters, linkage=r.linkage, connectivity=r.connectivity)
    else:
        raise RuntimeError('Unknown clusterer')
    return clstr


def bench_clustering(X_data, y_data, model, to_scale=False):
    if to_scale:
        estimator = make_pipeline(StandardScaler(), model).fit(X_data)
    else:
        estimator = make_pipeline(model).fit(X_data)

    clustering_metrics = {'homogeneity_score': homogeneity_score(y_data, estimator[-1].labels_),
                          'completeness_score': completeness_score(y_data, estimator[-1].labels_),
                          'v_measure_score': v_measure_score(y_data, estimator[-1].labels_),
                          'adjusted_rand_score': adjusted_rand_score(y_data, estimator[-1].labels_),
                          'adjusted_mutual_info_score': adjusted_mutual_info_score(y_data, estimator[-1].labels_),
                          'silhouette_score': silhouette_score(X_data, estimator[-1].labels_
                                                               )}

    return clustering_metrics


def run_bench_clustering(y_data, X_data, clstr_params):
    clstr_scores = []
    for model_name in clstr_params.keys():
        paramms = clstr_params[model_name]
        models = pd.DataFrame(paramms).apply(lambda x: init_clstr_model(model_name, x), axis=1)
        for model in models:
            model_results = bench_clustering(X_data, y_data, model)
            model_results['model'] = str(model)
            clstr_scores.append(model_results)
    return clstr_scores


def init_clf_model(model_name):
    if model_name == 'LR':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
    elif model_name == 'SVM':
        from sklearn.svm import SVC
        clf = SVC()
    elif model_name == 'LinearSVC':
        from sklearn.svm import LinearSVC
        clf = LinearSVC()
    elif model_name == 'SVM_ovr':
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier
        clf = OneVsRestClassifier(SVC())
    elif model_name == 'AB':
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier()
    elif model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier()
    elif model_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
    elif model_name == 'xgboost':
        from xgboost import XGBClassifier
        clf = XGBClassifier()
    elif model_name == 'mlpclassifier':
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier()
    else:
        raise RuntimeError('Unknown classifier')
    return clf


def clf_evaluate_models(X_train, y_train, X_test, y_test, model_params, n_splits=5, f1_averaged='micro',
                        scoring_func=None, debug=False):
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    best_clfs = {}
    scores = {'model': [], 'f1': [], 'precision': [], 'recall': [], 'train_accuracy': [], 'test_accuracy': [],
              'params': []}
    all_scores = {}
    model_names = list(model_params.keys())
    for model_name in model_names:
        if debug:
            print(f'Started evaluating {model_name}')
        model = init_clf_model(model_name)
        clf_steps = [(model_name, model)]

        pipe = Pipeline(steps=clf_steps)
        clf = GridSearchCV(pipe, model_params[model_name], scoring=scoring_func, cv=skf, n_jobs=-1)
        clf.fit(X_train, y_train)

        best_model = clf.best_estimator_[model_name]
        pred = best_model.predict(X_test)

        f1 = f1_score(y_test, pred, average=f1_averaged)
        score_train = best_model.score(X_train, y_train)
        score_test = best_model.score(X_test, y_test)
        precision = precision_score(y_test, pred, average=f1_averaged)
        recall = recall_score(y_test, pred, average=f1_averaged)

        scores['model'].append(model_name)
        scores['f1'].append(f1)
        scores['precision'].append(precision)
        scores['recall'].append(recall)
        scores['train_accuracy'].append(score_train)
        scores['test_accuracy'].append(score_test)
        scores['params'].append(clf.best_params_)
        all_scores[model_name] = clf.cv_results_
        if debug:
            print(f'Best params for {model_name}:', clf.best_params_)
            print(
                f'scores for the best model: f1 = {f1} precision = {precision} recall = {recall} train_accuracy = {score_train} test_accuracy = {score_test}')
            print()

        best_clfs[model_name] = clf.best_estimator_
    print(pd.DataFrame({'classifier': scores['model'],
                        'f1': scores['f1'],
                        'precision': scores['precision'],
                        'recall': scores['recall'],
                        'train_accuracy': scores['train_accuracy'],
                        'test_accuracy': scores['test_accuracy']
                        }).set_index('classifier').T)
    # best_model_idx = max(range(len(scores)), key=lambda i: scores[i])
    # print(f'Best model is {model_names[best_model_idx]} with params: {params[best_model_idx]}')

    return best_clfs, scores, all_scores
