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


#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def get_atom_env_feature(df):
    #a0 is always H, so only calculate for a1
    a1_env = {}

    a1_env['degree'] = []
    a1_env['explicit_valence'] = []
    a1_env['hybridization'] = []
    a1_env['implicit_valence'] = []
    a1_env['is_aromatic'] = []
    a1_env['mass'] = []
    a1_env['total_degree'] = []
    a1_env['total_hs'] = []
    a1_env['total_valence'] = []
    a1_env['in_ring'] = []
    a1_env['in_ring3'] = []
    a1_env['in_ring4'] = []
    a1_env['in_ring5'] = []
    a1_env['in_ring6'] = []
    a1_env['in_ring7'] = []
    a1_env['in_ring8'] = []

    cols = ['degree', 'explicit_valence', 'hybridization', 'implicit_valence', 'is_aromatic', 'mass',
            'total_degree', 'total_hs', 'total_valence', 'in_ring', 'in_ring3', 'in_ring4', 'in_ring5',
            'in_ring6', 'in_ring7', 'in_ring8']

    for idx, row in df.iterrows():
        try:
            a1 = row['mols'].GetAtomWithIdx(row['atom_index_1'])
          
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
      
    for col in cols:
        df['a1_' + col] = a1_env[col]

    ret_cols = [col for col in df.columns if 'a1' in col]

    return df, ret_cols

#https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization
def calc_atom_neighbor_feature(atom):
    feat = {}
    nb = [a for a in atom.GetNeighbors()] 
    nb_h = sum([a.GetSymbol() == 'H' for a in nb]) 
    nb_o = sum([a.GetSymbol() == 'O' for a in nb]) 
    nb_c = sum([a.GetSymbol() == 'C' for a in nb]) 
    nb_n = sum([a.GetSymbol() == 'N' for a in nb]) 
    nb_other = len(nb) - nb_h - nb_o - nb_n - nb_c

    feat['neighbor_count_h'] = nb_h
    feat['neighbor_count_o'] = nb_o
    feat['neighbor_count_c'] = nb_c
    feat['neighbor_count_n'] = nb_n
    feat['neighbor_count_other'] = nb_other
    feat['neighbor_degree_mean'] = np.mean([a.GetDegree() for a in nb])
    feat['neighbor_is_aromatic_mean'] = np.mean([a.GetIsAromatic() for a in nb]) 
    feat['neighbor_is_aromatic_count'] = sum([a.GetIsAromatic() for a in nb])
    feat['neighbor_in_ring_mean'] = np.mean([a.IsInRing() for a in nb])
    feat['neighbor_in_ring_count'] = sum([a.IsInRing() for a in nb])
    feat['neighbor_in_ring3_mean'] = np.mean([a.IsInRingSize(3) for a in nb]) 
    feat['neighbor_in_ring3_count'] = sum([a.IsInRingSize(3) for a in nb])
    feat['neighbor_in_ring4_mean'] = np.mean([a.IsInRingSize(4) for a in nb]) 
    feat['neighbor_in_ring4_count'] = sum([a.IsInRingSize(4) for a in nb])
    feat['neighbor_in_ring5_mean'] = np.mean([a.IsInRingSize(5) for a in nb])
    feat['neighbor_in_ring5_count'] = sum([a.IsInRingSize(5) for a in nb])
    feat['neighbor_in_ring6_mean'] = np.mean([a.IsInRingSize(6) for a in nb])
    feat['neighbor_in_ring6_count'] = sum([a.IsInRingSize(6) for a in nb])
    feat['neighbor_in_ring7_mean'] = np.mean([a.IsInRingSize(7) for a in nb])
    feat['neighbor_in_ring7_count'] = sum([a.IsInRingSize(7) for a in nb])
    feat['neighbor_in_ring8_mean'] = np.mean([a.IsInRingSize(8) for a in nb])
    feat['neighbor_in_ring8_count'] = sum([a.IsInRingSize(8) for a in nb])


    return feat

def get_atom_neighbor_feature(df):

    cols = ['neighbor_count_h', 'neighbor_count_o', 'neighbor_count_c', 'neighbor_count_n', 
            'neighbor_count_other', 'neighbor_degree_mean', 'neighbor_is_aromatic_mean', 'neighbor_is_aromatic_count',
            'neighbor_in_ring_mean', 'neighbor_in_ring_count', 'neighbor_in_ring3_mean', 'neighbor_in_ring3_count',
            'neighbor_in_ring4_mean', 'neighbor_in_ring4_count', 'neighbor_in_ring5_mean', 'neighbor_in_ring5_count',
            'neighbor_in_ring6_mean', 'neighbor_in_ring6_count', 'neighbor_in_ring7_mean', 'neighbor_in_ring7_count',
            'neighbor_in_ring8_mean', 'neighbor_in_ring8_count']

    for a in ['a0', 'a1']:
        for col in cols:
            df[a + '_' + col] = 0.0

    for idx, row in df.iterrows():
        try:
            #when mol is available, calculate neighbor feature for a0, a1
            a0 = row['mols'].GetAtomWithIdx(row['atom_index_0'])
            a1 = row['mols'].GetAtomWithIdx(row['atom_index_1'])

            for a in ['a0', 'a1']:
                if a == 'a0':
                    feat = calc_atom_neighbor_feature(a0)
                    for col in cols:
                        df.at[idx, a + '_' + col] = feat[col]
                else:
                    feat = calc_atom_neighbor_feature(a1)
                    for col in cols:
                        df.at[idx, a + '_' + col] = feat[col]                                 
        except:
            for a in ['a0', 'a1']:
                if a == 'a0':
                    for col in cols:
                        df.at[idx, a + '_' + col] = np.nan
                else:
                    for col in cols:
                        df.at[idx, a + '_' + col] = np.nan      
    ret_cols = [col for col in df.columns if ('a0' in col) or ('a1' in col)]  

    return df, ret_cols