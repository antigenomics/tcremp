import numpy as np

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import multilabel_confusion_matrix

import tcremp_legacy.ml_utils as ml_utils


def precision_recall_fscore(df, ytrue, ypred, label):
    labels = np.unique(ytrue)
    precisions = []  # per epitope
    recalls = []  # per epitope
    accuracies = []  # per epitope
    weights = []  # per epitope
    supports = []

    fn_per_epi = df[df['cluster'].isnull()][label].value_counts()  # Retain value counts for false negatives

    epmetrics = {'accuracy': {},
                 'precision': {},
                 'recall': {},
                 'f1-score': {},
                 'support': {}}
    for (i, cm) in enumerate(multilabel_confusion_matrix(ytrue,
                                                         ypred,
                                                         labels=labels)):  # per epitope
        # for order see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        lbl = labels[i]

        missing_fn = fn_per_epi.get(lbl, 0)
        fn += missing_fn

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if tp + tn == 0:
            accuracy = 0
        else:
            accuracy = (tp + tn) / (tp + tn + fp + fn)

        # weighting='average'
        support = sum(ytrue == lbl)
        w = support / ytrue.shape[0]
        weights.append(w)

        precision *= w
        recall *= w
        accuracy *= w

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        supports.append(support)

        epmetrics['accuracy'][labels[i]] = accuracy
        epmetrics['precision'][labels[i]] = precision
        epmetrics['recall'][labels[i]] = recall
        if (precision * recall == 0) | (precision + recall == 0):
            f = 0
        else:
            f = 2 * (precision * recall) / (precision + recall)

        epmetrics['f1-score'][labels[i]] = f
        epmetrics['support'][labels[i]] = support

    # deal with epitopes for which we had no prediction 
    uncalled_epis = set(df[label]).difference(labels)
    if len(uncalled_epis) != 0:
        print('No predictions made for: ', uncalled_epis)
    for i in uncalled_epis:
        accuracy = 0
        precision = 0
        recall = 0
        fscore = 0
        support = sum(ytrue == i)
        recalls.append(recall)
        precisions.append(precision)
        supports.append(sum(ytrue == i))

        epmetrics['accuracy'][i] = accuracy
        epmetrics['precision'][i] = precision
        epmetrics['recall'][i] = recall
        epmetrics['f1-score'][i] = fscore
        epmetrics['support'][i] = support

    recall = sum(recalls)
    precision = sum(precisions)
    support = sum(supports)

    if (precision * recall == 0) | (precision + recall == 0):
        f = 0
    else:
        f = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f, support, epmetrics


def get_clustermetrics(data_df, label):
    binom_res = data_df[['cluster', 'label_cluster', 'total_cluster', 'count_matched', 'fraction_matched', 'p_value',
                         'fraction_matched_exp', 'is_cluster']].drop_duplicates()

    # only TCRs in clusters
    df = data_df[data_df['is_cluster'] == 1]

    # Overall purity metrics
    purity = ml_utils.count_clstr_purity(binom_res)

    # Compute predictive metrics
    ypred = df['label_cluster']
    ytrue = df[label]
    ami = adjusted_mutual_info_score(ytrue, ypred)
    accuracy, precision, recall, f1score, support, epmetrics = precision_recall_fscore(
        df, ytrue, ypred, label)
    ami = adjusted_mutual_info_score(ytrue, ypred)

    counts = {k: v for k, v in df[label].value_counts().reset_index().values.tolist()}
    maincluster = {ep: df[df[label] == ep]['cluster'].value_counts().index[0] for ep in
                   df[label].unique()}  # Find the largest cluster per epitope

    consistencymap = {ep: len(df[(df[label] == ep) & (df['cluster'] == maincluster[ep])]) / counts[ep] for ep in
                      counts.keys()}  # Get consistency scores per epitope

    return {'purity': round(purity, 2),
            # 'purity': np.mean(list(binom_res[binom_res['is_cluster']==1]['fraction_matched'])),
            # Purity of all clusters weighted equally (frequency)

            'retention': round(len(data_df[data_df['is_cluster'] == 1]) / len(data_df), 2),
            # Proportion of clustered TCRs
            'consistency': round(np.mean([(consistencymap[ep] * counts[ep]) / len(df) for ep in consistencymap.keys()]),
                                 4),  # Proportion of an epitope assigned to a given cluster
            'ami': round(ami, 2),  # Adjusted mutual information
            # 'accuracy':accuracy,    # Balanced accuracy
            'precision': round(precision, 2),  # Precision over all epitopes
            'recall': round(recall, 2),  # Recall over all epitopes
            'f1-score': round(f1score, 2),  # F1 over all epitopes
            # 'support':support,  # Support
            'mean_clustsize': round(np.mean(list(binom_res[binom_res['is_cluster'] == 1]['total_cluster'])), 2),
            # Average cluster size
            }
