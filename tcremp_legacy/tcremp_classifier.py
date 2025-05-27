import pandas as pd

import sys

sys.path.append("../")
sys.path.append("mirpy/")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import tcremp_legacy.ml_utils as ml_utils


class TCRemP_clf:
    def __init__(self, model_name):
        self.model_name = model_name
        self.annotation_id = 'annotId'
        self.label = {}
        self.__max_depth = 20
        self.y_train = {}
        self.y_test = {}
        self.X_train = {}
        self.X_test = {}
        self.model = {}
        self.test_pred = {}
        self.test_pred_proba = {}
        self.clsf_metrics = {}
        self.roc_auc_proba_df = {}
        self.roc_auc_proba = {}

    def clf(self, chain, data, label_cl, model=None, test_size=0.3, pca_sep_chains=False):
        if model is None:
            model = RandomForestClassifier(max_depth=self.__max_depth, random_state=7)
        self.model[chain] = model
        self.label[chain] = label_cl
        y_data = data.annot[chain][label_cl]
        if pca_sep_chains:
            X_data = data.pca_ad['all'][chain].drop(self.annotation_id, axis=1, errors='ignore')
        else:
            X_data = data.pca[chain].drop(self.annotation_id, axis=1, errors='ignore')
        self.__clf_model(chain, y_data, X_data, test_size)

    def __clf_model(self, chain, y_data, X_data, test_size):

        self.y_train[chain], self.y_test[chain], self.X_train[chain], self.X_test[chain] = train_test_split(y_data,
                                                                                                            X_data,
                                                                                                            test_size=test_size)

        self.model[chain].fit(self.X_train[chain], self.y_train[chain])

        self.test_pred[chain] = self.model[chain].predict(self.X_test[chain])
        self.test_pred_proba[chain] = self.model[chain].predict_proba(self.X_test[chain])

        self.clsf_metrics[chain] = {}
        self.clsf_metrics[chain]['train_acc'] = self.model[chain].score(self.X_train[chain], self.y_train[chain])
        self.clsf_metrics[chain]['test_acc'] = self.model[chain].score(self.X_test[chain], self.y_test[chain])
        self.clsf_metrics[chain]['f1_macro'] = ml_utils.clsf_metrics(self.test_pred[chain], self.y_test[chain], 'macro')
        self.clsf_metrics[chain]['f1_weighted'] = ml_utils.clsf_metrics(self.test_pred[chain], self.y_test[chain],
                                                                        'weighted')

        print(f"Train accuracy: {self.clsf_metrics[chain]['train_acc']}")
        print(f"Test accuracy: {self.clsf_metrics[chain]['test_acc']}")
        print(f"macro: {self.clsf_metrics[chain]['f1_macro']}")
        print(f"weighted: {self.clsf_metrics[chain]['f1_weighted']}")

    def roc_auc(self, chain, ax=None, show_legend=True, custom_palette=None):
        classes_list = list(self.model[chain].classes_)
        y_test_curv = label_binarize(self.y_test[chain], classes=classes_list)

        roc_auc_proba = ml_utils.roc_auc_count(y_test_curv, self.test_pred_proba[chain])
        self.roc_auc_proba[chain] = roc_auc_proba
        self.roc_auc_proba_df[chain] = pd.DataFrame({'class': classes_list, 'roc_auc': roc_auc_proba.values()})

        ml_utils.plot_roccurve_multi(classes_list, y_test_curv, self.test_pred_proba[chain], f'ROC curves, {chain}', ax,
                                     custom_palette=custom_palette, test_acc=self.clsf_metrics[chain]['test_acc'],
                                     f1_weighted=self.clsf_metrics[chain]['f1_weighted'], show_legend=show_legend)


class TCRemp_clf_pred(TCRemP_clf):
    def __init__(self, model_name):
        TCRemP_clf.__init__(self, model_name)
        self.annotation_id = 'annotId'
        self.X_pred = {}
        self.pred = {}
        self.y_pred_real = {}
        self.pred_proba = {}
        self.clsf_metrics_pred = {}
        self.roc_auc_proba_pred_df = {}
        self.pca_sep_chains = None

    def clf(self, chain, data, label_cl, model=None, test_size=0.3, pca_sep_chains=False):
        self.pca_sep_chains = pca_sep_chains
        if model is None:
            model = RandomForestClassifier(max_depth=self.__max_depth, random_state=7)
        self.model[chain] = model

        y_data = data.annot[chain][data.annot[chain]['data_type'] == 'train'][label_cl]
        if self.pca_sep_chains:
            X_data = data.pca_ad['all'][chain].merge(data.annot[chain][[self.annotation_id, 'data_type']])
        else:
            X_data = data.pca[chain].merge(data.annot[chain][[self.annotation_id, 'data_type']])

        X_data = X_data[X_data['data_type'] == 'train'].drop([self.annotation_id, 'data_type'], axis=1, errors='ignore')
        self.__clf_model(chain, y_data, X_data, test_size)

    def clf_pred(self, chain, data, label_cl=None):

        if self.pca_sep_chains:
            X_pred = data.pca_ad['all'][chain].merge(data.annot[chain][[self.annotation_id, 'data_type']])
        else:
            X_pred = data.pca[chain].merge(data.annot[chain][[self.annotation_id, 'data_type']])

        self.X_pred[chain] = X_pred[X_pred['data_type'] == 'pred'].drop([self.annotation_id, 'data_type'], axis=1,
                                                                        errors='ignore')
        self.pred[chain] = self.model[chain].predict(self.X_pred[chain])
        self.pred_proba[chain] = self.model[chain].predict_proba(self.X_pred[chain])

        self.clsf_metrics_pred[chain] = {}

        if label_cl is None:
            self.y_pred_real[chain] = None
        else:
            self.y_pred_real[chain] = data.annot[chain][data.annot[chain]['data_type'] == 'pred'][label_cl]
            self.clsf_metrics_pred[chain]['pred_acc'] = self.model[chain].score(self.X_pred[chain],
                                                                                self.y_pred_real[chain])
            self.clsf_metrics_pred[chain]['f1_macro'] = ml_utils.clsf_metrics(self.pred[chain], self.y_pred_real[chain],
                                                                              'macro')
            self.clsf_metrics_pred[chain]['f1_weighted'] = ml_utils.clsf_metrics(self.pred[chain],
                                                                                 self.y_pred_real[chain], 'weighted')

            print(f"Prediction accuracy: {self.clsf_metrics_pred[chain]['pred_acc']}")
            print(f"macro: {self.clsf_metrics[chain]['f1_macro']}")
            print(f"weighted: {self.clsf_metrics[chain]['f1_weighted']}")

    def roc_auc_pred(self, chain, ax=None, show_legend=True, custom_palette=None):
        classes_list = list(self.model[chain].classes_)
        y_real_curv = label_binarize(self.y_pred_real[chain], classes=classes_list)

        roc_auc_proba = ml_utils.roc_auc_count(y_real_curv, self.pred_proba[chain])
        self.roc_auc_proba_pred_df[chain] = pd.DataFrame({'class': classes_list, 'roc_auc': roc_auc_proba.values()})

        ml_utils.plot_roccurve_multi(classes_list, y_real_curv, self.pred_proba[chain], f'ROC curves, {chain}', ax,
                                     custom_palette=custom_palette, show_legend=show_legend)
