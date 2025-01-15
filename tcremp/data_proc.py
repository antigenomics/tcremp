import logging

import pandas as pd
import math

import warnings

warnings.filterwarnings("ignore")


def write_filtered_out(df, file_dir, header=None):
    f_path = f'{file_dir}/filtered_out_data.txt'
    if len(df[df['filtered_out'] == True]) > 0:
        with open(f_path, "a") as f:
            if header:
                f.write(f"\n{header}\n")
        df[df['filtered_out'] == True].reset_index(drop=True).drop('filtered_out', axis=1).to_csv(f_path, sep='\t',
                                                                                                  index=False, mode='a')


def check_columns(data, chain, tcr_columns):
    def validate_inner(column_list):
        if not set(column_list).issubset(data.columns):
            raise Exception(f'Incorrect columns names or any column is absent. List of required columns is: {column_list}')

    if chain == 'TRA_TRB':
        validate_inner(tcr_columns['TRA'] + tcr_columns['TRB'])
    else:
        validate_inner(tcr_columns[chain])


def clean_at_least_cdr3a_or_cdr3b(data, cdr3a, cdr3b, file_dir=None):
    df = data.copy()
    df['filtered_out'] = df[cdr3a].isna() * df[cdr3b].isna()
    if df['filtered_out'].sum() > 0:
        logging.warning(f"filtered out {df['filtered_out'].sum()} rows from input file as both TCR alpha and beta "
                        f"info are missing")
    if file_dir:
        write_filtered_out(df, file_dir, 'Both a_cdr3aa and b_cdr3aa are None')
    return df[~df['filtered_out']].reset_index(drop=True).drop('filtered_out', axis=1)


# Data processing
# work on filter_clones_data
def annot_id(data, annotation_id_str):
    df = data.copy()
    df[annotation_id_str] = df.index
    return df


def remove_asterisk(data, tcr_columns):
    df = data.copy()
    df[tcr_columns[1]] = df[tcr_columns[1]].str.split('*', n=1, expand=True)[0]
    df[tcr_columns[2]] = df[tcr_columns[2]].str.split('*', n=1, expand=True)[0]
    return df


def add_allele(data, tcr_columns):
    df = data.copy()
    df[tcr_columns[1]] = df[tcr_columns[1]] + '*01'
    df[tcr_columns[2]] = df[tcr_columns[2]] + '*01'
    return df


def remove_backslash(data, tcr_columns):
    df = data.copy()
    df[tcr_columns[1]] = df[tcr_columns[1]].str.replace('/', '')
    df[tcr_columns[2]] = df[tcr_columns[2]].str.replace('/', '')
    return df


def filter_clones_data(df_clones, tcr_columns, file_dir=None, cdr3nt=None):
    df = df_clones.copy()
    df['filtered_out'] = df[tcr_columns[0]].isna() + df[tcr_columns[0]].str.contains(',') + df[
        tcr_columns[0]].str.contains('\.') + df[tcr_columns[0]].str.contains('\_') + df[tcr_columns[0]].str.contains(
        '\*') + df[tcr_columns[1]].isna() + df[tcr_columns[1]].str.contains(',') + df[tcr_columns[1]].str.contains(
        '\.') + df[tcr_columns[2]].isna() + df[tcr_columns[2]].str.contains(',') + df[tcr_columns[2]].str.contains('\.')
    if cdr3nt:
        df['filtered_out'] = df['filtered_out'] + df[tcr_columns[0]].isna() + df[cdr3nt].str.contains(',') + df[
            cdr3nt].str.contains('\.') + df[cdr3nt].str.contains('\_') + df[cdr3nt].str.contains('\*')

    if file_dir:
        write_filtered_out(df, file_dir, 'CDR3 or V or J is absent or contains invalid character')

    return df[df['filtered_out'] == 0].reset_index(drop=True).drop('filtered_out', axis=1)


def filter_segments(df_clones, segments_path='mirpy/mir/resources/segments.txt', v='v', j='j', organism='HomoSapiens',
                    file_dir=None, chain=['TRA', 'TRB']):
    segs = pd.read_csv(segments_path, sep='\t')
    segs = segs[segs['organism'] == organism]
    segs = segs[segs['id'].apply(lambda x: x[:3] in chain)]
    segs_ids = list(segs['id'].drop_duplicates())
    df = df_clones.copy()
    df['filtered_out'] = -df[v].isin(segs_ids) + df[v].isna() + -df[j].isin(segs_ids) + df[j].isna()

    if sum(df['filtered_out']) > 0:
        logging.warn(f"Have filtered out {sum(df['filtered_out'])} due to incorrect segments")
    if file_dir:
        write_filtered_out(df, file_dir, 'V or J segment is not present in resource segments list for this species:')

    return df[~df['filtered_out']].reset_index(drop=True).drop('filtered_out', axis=1)


def freq_labels(label, data_id, data_preped, n=5, tr=3):
    df = data_preped.copy()
    label_counts = data_preped.groupby(label)[data_id].count().reset_index(name='counts').sort_values('counts',
                                                                                                      ascending=False)
    labels_freq = label_counts.head(n)
    labels_freq = list(labels_freq[labels_freq['counts'] >= tr][label])
    df[str(label + '_freq')] = df[label].apply(lambda x: x if x in labels_freq else 'other')
    return df


def freq_labels_tr_list(label, data_id, data_preped, freqs_list):
    df = data_preped.copy()
    label_counts = df.groupby(label)[data_id].count().sort_values().reset_index(name='counts')
    label_count = {}
    for i in freqs_list:
        label_count[str(label + '_freq_' + str(i))] = i
    label_high_count_dict = {}
    for k in label_count.keys():
        label_high_count_dict[k] = list(label_counts[label_counts['counts'] >= label_count[k]][label])
        df[k] = df[label].apply(lambda x: x if x in label_high_count_dict[k] else 'other')
    return df


def filter_save_freq_subsets(data, chain, label, samples_n, freq_col_list, dataset_outputs_suf):
    for i in samples_n:
        v_output_path = f'{dataset_outputs_suf}_{chain}_V{i}.csv'
        col = f'{label}_freq_{i}'
        df = data[data[col] != 'other'].drop(freq_col_list, axis=1)
        df.to_csv(v_output_path, sep='\t', index=False)


def filter_save_freq_subsets_paired(data, chain, label, samples_n, freq_col_list, dataset_outputs_suf):
    for i in samples_n:
        if chain == 'paired':
            v_output_path = f'{dataset_outputs_suf}_V{i}.csv'
        else:
            v_output_path = f'{dataset_outputs_suf}_{chain}_V{i}.csv'
        col = f'{label}_freq_{i}'
        df = data[data[col] != 'other'].drop(freq_col_list, axis=1)
        df.to_csv(v_output_path, sep='\t', index=False)


# 10x proc
def read_barcodes(barcodes_file):
    barcodes = pd.read_csv(barcodes_file, sep='\t', header=None)
    barcodes.columns = ['barcode']
    barcodes.index += 1
    barcodes['barcode_id'] = barcodes.index
    return barcodes


def read_features(features_file):
    features = pd.read_csv(features_file, sep='\t', header=None)
    features.index += 1
    features.columns = ['feature_code', 'value', 'type']
    features['feature_id'] = features.index
    features['feature_id'] = pd.to_numeric(features['feature_id'])
    return features


def read_matrix(matrix_file):
    matrix = pd.read_csv(matrix_file, sep='\t')
    matrix = matrix.drop([0])
    matrix[['feature_id', 'barcode_id', 'count']] = matrix[
        '%%MatrixMarket matrix coordinate integer general'].str.split(expand=True)
    matrix = matrix.drop(['%%MatrixMarket matrix coordinate integer general'], axis=1)
    matrix = matrix.apply(pd.to_numeric)
    matrix['count'] = matrix['count'].astype(int)
    return matrix


def get_value_matrix(matrix):
    v_t = matrix['value'].str.split('_', n=1, expand=True)
    matrix['value'] = v_t[0]
    matrix['value_type'] = v_t[1]
    return matrix


def get_barcode_top_tetramer(matrix):
    matrix = matrix.sort_values(by=['count'], ascending=False)
    tetramers = matrix.drop_duplicates('barcode')
    tetramers['top_tetramer'] = tetramers['value']
    return tetramers[['barcode', 'top_tetramer', 'count']]


def norm_logp(data, count_col):
    data[count_col] = data[count_col].apply(lambda x: math.log1p(x))
    return data


def pivot_data(data):
    data = data[['count', 'barcode', 'tetramer']]
    data = data.pivot_table('count', 'barcode', 'tetramer')
    data = data.fillna(0)
    return data
