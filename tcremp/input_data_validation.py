import pandas as pd
import tcremp.data_proc as data_proc


def validate_prototype_files(p_alpha_file, p_beta_file, p_file, chain, segments_path, prototypes_path,
                             cdr3aa_column='cdr3aa', cdr3nt_column=None, v_column='v', j_column='j'):
    if p_file is not None and (p_alpha_file is not None or p_beta_file is not None):
        raise ValueError('You should specify iether -p separately from -p_a -p_b params')
    if p_file is not None:
        prototypes = pd.read_csv(p_file, sep='\t')
        if chain == 'TRA_TRB':
            assert 'chain' in prototypes.columns
        elif chain == 'TRA':
            prototypes['chain'] = 'TRA'
        elif chain == 'TRB':
            prototypes['chain'] = 'TRB'
    else:
        all_prototypes = []
        if p_alpha_file is not None:
            alpha_prototypes = pd.read_csv(p_alpha_file, sep='\t')
            alpha_prototypes['chain'] = 'TRA'
            all_prototypes.append(alpha_prototypes)
        if p_beta_file is not None:
            beta_prototypes = pd.read_csv(p_beta_file, sep='\t')
            beta_prototypes['chain'] = 'TRB'
            all_prototypes.append(beta_prototypes)
        prototypes = pd.concat(all_prototypes)

    prototypes[v_column] = prototypes[v_column].apply(lambda x: x + '*01' if '*' not in x else x)
    prototypes[j_column] = prototypes[j_column].apply(lambda x: x + '*01' if '*' not in x else x)

    prototypes = data_proc.filter_clones_data(prototypes, [cdr3aa_column, v_column, j_column],
                                              cdr3nt=cdr3nt_column)

    prototypes = data_proc.filter_segments(prototypes, segments_path=segments_path, v=v_column, j=j_column)
    prototypes_count = 0

    prototypes_a = prototypes[prototypes['chain'] == 'TRA']
    if len(prototypes_a) > 0:
        prototypes_a.reset_index(drop=True).drop('chain', axis=1).to_csv(prototypes_path['TRA'], sep='\t')
        prototypes_count = len(prototypes_a)

    prototypes_b = prototypes[prototypes['chain'] == 'TRB']
    if len(prototypes_b) > 0:
        prototypes_b.reset_index(drop=True).drop('chain', axis=1).to_csv(prototypes_path['TRB'], sep='\t')
        prototypes_count += len(prototypes_b)

    return prototypes_count
