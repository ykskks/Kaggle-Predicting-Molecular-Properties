import time
from contextlib import contextmanager
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import feather
from lightgbm.callback import _format_eval_result 

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'{name}: finished in {end_time - start_time} s')


def load_datasets(feats):
    train_feats = [feather.read_dataframe(f'features/{feat}_train.feather') for feat in feats]
    train = pd.concat(train_feats, axis=1)
    test_feats = [feather.read_dataframe(f'features/{feat}_test.feather') for feat in feats]
    test = pd.concat(test_feats, axis=1)
    return train, test


def load_target(target_name):
    train = feather.read_dataframe('data/input/train.feather')
    target = train[target_name]
    return target


def get_categorical_feats(feats):
    categorical_feats = []
    train, test = load_datasets(feats)

    for col in train.columns:
        categorical_feats.append(col)

    return categorical_feats


def calculate_metric(predicted_values, true_values, coupling_types, floor=1e-9):
    """Inputs should be in numpy.array format"""
    metric = 0
    for coupling_type in np.unique(coupling_types):
        group_metric = np.log(mean_absolute_error(true_values[coupling_types == coupling_type], predicted_values[coupling_types == coupling_type]))
        metric += max(group_metric, floor)
    return metric / len(np.unique(coupling_types))


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

def get_atom_env_feature(df):
    a0_env, a1_env = {}, {}

    a0_env['degree'], a1_env['degree'] = [], []
    a0_env['explicit_valence'], a1_env['explicit_valence'] = [], []
    a0_env['hybridization'], a1_env['hybridization'] = [], []
    a0_env['implicit_valence'], a1_env['implicit_valence'] = [], []
    a0_env['is_aromatic'], a1_env['is_aromatic'] = [], []
    a0_env['mass'], a1_env['mass'] = [], []
    a0_env['total_degree'], a1_env['total_degree'] = [], []
    a0_env['total_hs'], a1_env['total_hs'] = [], []
    a0_env['total_valence'], a1_env['total_valence'] = [], []
    a0_env['in_ring'], a1_env['in_ring'] = [], []
    a0_env['in_ring3'], a1_env['in_ring3'] = [], []
    a0_env['in_ring4'], a1_env['in_ring4'] = [], []
    a0_env['in_ring5'], a1_env['in_ring5'] = [], []
    a0_env['in_ring6'], a1_env['in_ring6'] = [], []
    a0_env['in_ring7'], a1_env['in_ring7'] = [], []
    a0_env['in_ring8'], a1_env['in_ring8'] = [], []

    cols = ['degree', 'explicit_valence', 'hybridization', 'implicit_valence', 'is_aromatic', 'mass',
            'total_degree', 'total_hs', 'total_valence', 'in_ring', 'in_ring3', 'in_ring4', 'in_ring5',
            'in_ring6', 'in_ring7', 'in_ring8']

    for idx, row in df.iterrows():
        try:
            a0 = row['mols'].GetAtomWithIdx(row['atom_index_0'])
            a1 = row['mols'].GetAtomWithIdx(row['atom_index_1'])

            a0_env['degree'].append(a0.GetDegree())
            a0_env['explicit_valence'].append(a0.GetExplicitValence())
            a0_env['hybridization'].append(a0.GetHybridization())
            a0_env['implicit_valence'].append(a0.GetImplicitValence())
            a0_env['is_aromatic'].append(a0.GetIsAromatic())
            a0_env['mass'].append(a0.GetMass())
            a0_env['total_degree'].append(a0.GetTotalDegree())
            a0_env['total_hs'].append(a0.GetTotalNumHs())
            a0_env['total_valence'].append(a0.GetTotalValence())
            a0_env['in_ring'].append(a0.IsInRing())
            a0_env['in_ring3'].append(a0.IsInRingSize(3))
            a0_env['in_ring4'].append(a0.IsInRingSize(4))
            a0_env['in_ring5'].append(a0.IsInRingSize(5))
            a0_env['in_ring6'].append(a0.IsInRingSize(6))
            a0_env['in_ring7'].append(a0.IsInRingSize(7))
            a0_env['in_ring8'].append(a0.IsInRingSize(8))

            a1_env['degree'].append(a1.GetDegree())
            a1_env['explicit_valence'].append(a1.GetExplicitValence())
            a1_env['hybridization'].append(a1.GetHybridization())
            a1_env['implicit_valence'].append(a1.GetImplicitValence())
            a1_env['is_aromatic'].append(a1.GetIsAromatic())
            a1_env['mass'].append(a1.GetMass())
            a1_env['total_degree'].append(a1.GetTotalDegree())
            a1_env['total_hs'].append(a1.GetTotalNumHs())
            a1_env['total_valence'].append(a1.GetTotalValence())
            a1_env['in_ring'].append(a1.IsInRing())
            a1_env['in_ring3'].append(a1.IsInRingSize(3))
            a1_env['in_ring4'].append(a1.IsInRingSize(4))
            a1_env['in_ring5'].append(a1.IsInRingSize(5))
            a1_env['in_ring6'].append(a1.IsInRingSize(6))
            a1_env['in_ring7'].append(a1.IsInRingSize(7))
            a1_env['in_ring8'].append(a1.IsInRingSize(8))

        except:
            a0_env['degree'].append(np.nan)
            a0_env['explicit_valence'].append(np.nan)
            a0_env['hybridization'].append(np.nan)
            a0_env['implicit_valence'].append(np.nan)
            a0_env['is_aromatic'].append(np.nan)
            a0_env['mass'].append(np.nan)
            a0_env['total_degree'].append(np.nan)
            a0_env['total_hs'].append(np.nan)
            a0_env['total_valence'].append(np.nan)
            a0_env['in_ring'].append(np.nan)
            a0_env['in_ring3'].append(np.nan)
            a0_env['in_ring4'].append(np.nan)
            a0_env['in_ring5'].append(np.nan)
            a0_env['in_ring6'].append(np.nan)
            a0_env['in_ring7'].append(np.nan)
            a0_env['in_ring8'].append(np.nan)

            a1_env['degree'].append(np.nan)
            a1_env['explicit_valence'].append(np.nan)
            a1_env['hybridization'].append(np.nan)
            a1_env['implicit_valence'].append(np.nan)
            a1_env['is_aromatic'].append(np.nan)
            a1_env['mass'].append(np.nan)
            a1_env['total_degree'].append(np.nan)
            a1_env['total_hs'].append(np.nan)
            a1_env['total_valence'].append(np.nan)
            a1_env['in_ring'].append(np.nan)
            a1_env['in_ring3'].append(np.nan)
            a1_env['in_ring4'].append(np.nan)
            a1_env['in_ring5'].append(np.nan)
            a1_env['in_ring6'].append(np.nan)
            a1_env['in_ring7'].append(np.nan)
            a1_env['in_ring8'].append(np.nan)
      

    for a in ['a0', 'a1']:
        for col in cols:
            if a == 'a0':
                df[a + '_' + col] = a0_env[col]
            else:
                df[a + '_' + col] = a1_env[col]

    cols = [col for col in df.columns if ('a0' in col) or ('a1' in col)]

    return df, cols