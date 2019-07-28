import sys
import os
sys.path.append('.')
import re
import gc
import pickle
import math

import pandas as pd
import numpy as np
import feather
#from rdkit.Chem import Descriptors, Descriptors3D, MolFromMolBlock, MACCSkeys, DataStructs
from tqdm import tqdm
tqdm.pandas()
from joblib import Parallel, delayed
import openbabel

from features.base import get_arguments, get_features, generate_features, Feature
from utils import get_atom_env_feature, get_atom_neighbor_feature, calc_atom_neighbor_feature, reduce_mem_usage
# from utils import generate_brute_force_features

Feature.base_dir = 'features'


class CouplingType(Feature):
    def create_features(self):
        self.train['type'] = train['type']
        self.test['type'] = test['type']


class AtomPosition(Feature):
    def create_features(self):
        structures = feather.read_dataframe('./data/input/structures.feather')
        global train, test

        def map_atom_info(df, atom_idx):
            df = pd.merge(df, structures, how='left', left_on=['molecule_name', f'atom_index_{atom_idx}'], 
                            right_on=['molecule_name', 'atom_index'])
            df = df.drop('atom_index', axis=1)
            df = df.rename(columns={'x': f'x_{atom_idx}',
                                    'y': f'y_{atom_idx}',
                                    'z': f'z_{atom_idx}'})
            return df

        train = map_atom_info(train, 0)
        train = map_atom_info(train, 1)
        test = map_atom_info(test, 0)
        test = map_atom_info(test, 1)   

        pos_list = [i+'_'+j for i in ['x', 'y', 'z'] for j in ['0', '1']]
        for pos in pos_list:
            self.train[pos] = train[pos]
            self.test[pos] = test[pos]


class Atom(Feature):
    def create_features(self):
        structures = feather.read_dataframe('./data/input/structures.feather')
        global train, test

        def map_atom_info(df, atom_idx):
            df = pd.merge(df, structures, how='left', left_on=['molecule_name', f'atom_index_{atom_idx}'],
                          right_on=['molecule_name', 'atom_index'])
            df = df.drop('atom_index', axis=1)
            df = df.rename(columns={'atom': f'atom_{atom_idx}'})
            return df

        train = map_atom_info(train, 0)
        train = map_atom_info(train, 1)
        test = map_atom_info(test, 0)
        test = map_atom_info(test, 1)   

        self.train['atom_0'] = train['atom_0']
        self.train['atom_1'] = train['atom_1']
        self.test['atom_0'] = test['atom_0']
        self.test['atom_1'] = test['atom_1']


class AtomDistance(Feature):
    def create_features(self):
        structures = feather.read_dataframe('./data/input/structures.feather')
        global train, test

        radius = dict(zip(['H','C','N','O','F'], [0.38,0.77,0.75,0.73,0.71]))
        #en = dict(zip(['H','C','N','O','F'], [2.2,2.55,3.04,3.44,3.98]))

        def map_atom_info(df, atom_idx):
            df = pd.merge(df, structures, how='left', left_on=['molecule_name', f'atom_index_{atom_idx}'],
                          right_on=['molecule_name', 'atom_index'])
            df = df.drop('atom_index', axis=1)
            df = df.rename(columns={'x': f'x_{atom_idx}',
                                    'y': f'y_{atom_idx}',
                                    'z': f'z_{atom_idx}'})
            return df

        train = map_atom_info(train, 0)
        train = map_atom_info(train, 1)
        test = map_atom_info(test, 0)
        test = map_atom_info(test, 1)

        train_p_0 = train[['x_0', 'y_0', 'z_0']].values
        train_p_1 = train[['x_1', 'y_1', 'z_1']].values
        test_p_0 = test[['x_0', 'y_0', 'z_0']].values
        test_p_1 = test[['x_1', 'y_1', 'z_1']].values

        train['atom_distance'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        test['atom_distance'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

        def get_more_dist_features(df):

            tmp_0 = pd.DataFrame(df.groupby(['molecule_name', 'atom_index_0'])['atom_distance'].apply(lambda x: (1 / x ** 3))).rename(columns={'atom_distance': 'dist_inv_0'})
            tmp_1 = pd.DataFrame(df.groupby(['molecule_name', 'atom_index_1'])['atom_distance'].apply(lambda x: (1 / x ** 3))).rename(columns={'atom_distance': 'dist_inv_1'})

            tmp_0['molecule_name'], tmp_0['atom_index_0'] = df['molecule_name'], df['atom_index_0']
            tmp_1['molecule_name'], tmp_1['atom_index_1'] = df['molecule_name'], df['atom_index_1']

            tmp_0 = tmp_0.groupby(['molecule_name', 'atom_index_0'])['dist_inv_0'].sum().reset_index().rename(columns={'dist_inv_0':'dist_inv_0_sum'})
            tmp_1 = tmp_1.groupby(['molecule_name', 'atom_index_1'])['dist_inv_1'].sum().reset_index().rename(columns={'dist_inv_1':'dist_inv_1_sum'})

            df = df.merge(tmp_0, on=['molecule_name', 'atom_index_0'], how='left')
            df = df.merge(tmp_1, on=['molecule_name', 'atom_index_1'], how='left')
            df['dist_inv_sum_pair'] = df['dist_inv_0_sum']*df['dist_inv_1_sum'] / (df['dist_inv_0_sum']+df['dist_inv_1_sum'])

            # some more dist related feature using radius 
        
            df['radius_0'] = df['atom_x'].map(radius)
            #df['en_0'] = df['atom_x'].map(en)
            df['radius_1'] = df['atom_y'].map(radius)
            #df['en_1'] = df['atom_y'].map(en)

            df['bond_length'] = df['atom_distance'] - df['radius_0'] - df['radius_1']

            tmp_0 = df.groupby(['molecule_name', 'atom_index_0'])['bond_length'].agg(['mean', 'max', 'min', 'sum']).reset_index()
            tmp_0.columns = ['molecule_name', 'atom_index_0', 'bond_length_mean', 'bond_length_max', 'bond_length_min', 'bond_length_sum']
            tmp_1 = df.groupby(['molecule_name', 'atom_index_1'])['bond_length'].agg(['mean', 'max', 'min', 'sum']).reset_index()
            tmp_1.columns = ['molecule_name', 'atom_index_1', 'bond_length_mean', 'bond_length_max', 'bond_length_min', 'bond_length_sum']

            df = df.merge(tmp_0, on=['molecule_name', 'atom_index_0'], how='left')
            df = df.merge(tmp_1, on=['molecule_name', 'atom_index_1'], how='left', suffixes=['_a0', '_a1'])

            df['bond_length_mean_pair'] = df['bond_length_mean_a0'] + df['bond_length_mean_a1']
            df['bond_length_max_pair'] = df['bond_length_max_a0'] + df['bond_length_max_a1']
            df['bond_length_min_pair'] = df['bond_length_min_a0'] + df['bond_length_min_a1']
            df['bond_length_sum_pair'] = df['bond_length_sum_a0'] + df['bond_length_sum_a1']
            
            return df

        train = get_more_dist_features(train)
        test = get_more_dist_features(test)

        cols = ['atom_distance', 'dist_inv_0_sum', 'dist_inv_1_sum', 'dist_inv_sum_pair', 
                'bond_length_mean_a0', 'bond_length_max_a0', 'bond_length_min_a0', 'bond_length_sum_a0',
                'bond_length_mean_a1', 'bond_length_max_a1', 'bond_length_min_a1', 'bond_length_sum_a1',
                'bond_length_mean_pair', 'bond_length_max_pair', 'bond_length_min_pair', 'bond_length_sum_pair']

        for col in cols:
            self.train[col] = train[col]
            self.test[col] = test[col]


class RdkitDescriptors(Feature):
    def create_features(self):
        global train, test

        #load in mol files
        with open('./data/input/mols_without_Hs.pickle', 'rb') as f:
            rdkit_desc_df = pickle.load(f)

        #store functions in Descriptor, Descriptor3D modules in lists
        desc_2Ds = [v for k, v in Descriptors.__dict__.items() if not '__' in k and not bool(re.match('_', k)) and callable(v)]
        desc_3Ds = [v for k, v in Descriptors3D.__dict__.items() if not '__' in k and not bool(re.match('_', k)) and callable(v)]
        desc_2Ds_cols = [k for k, v in Descriptors.__dict__.items() if not '__' in k and not bool(re.match('_', k)) and callable(v)]
        desc_3Ds_cols = [k for k, v in Descriptors3D.__dict__.items() if not '__' in k and not bool(re.match('_', k)) and callable(v)]
        desc_cols = desc_2Ds_cols + desc_3Ds_cols

        #function for skipping molecules with no mol file
        def skip_nan(func, mol):
            try:
                return func(mol)
            except:
                return np.nan
        
        for desc_2D, desc_2Ds_col in zip(desc_2Ds, desc_2Ds_cols):
            rdkit_desc_df[desc_2Ds_col] = rdkit_desc_df['mols'].apply(lambda mol: skip_nan(desc_2D, mol))
        for desc_3D, desc_3Ds_col in zip(desc_3Ds, desc_3Ds_cols):
            rdkit_desc_df[desc_3Ds_col] = rdkit_desc_df['mols'].apply(lambda mol: skip_nan(desc_3D, mol))

        #merge with train, test dataset
        train = train.merge(rdkit_desc_df, left_on='molecule_name', right_on='ids', how='left')
        test = test.merge(rdkit_desc_df, left_on='molecule_name', right_on='ids', how='left')

        self.train[desc_cols] = train[desc_cols]
        self.test[desc_cols] = test[desc_cols]


class MaccsKey(Feature):
    def create_features(self):
        global train, test

        #load in mol files
        with open('./data/input/mols_without_Hs.pickle', 'rb') as f:
            maccs_df = pickle.load(f)

        BIT_LENGTH = 167
        maccs_keys = np.zeros((0, BIT_LENGTH), dtype=np.int8)
        for i in range(len(maccs_df)):
            try:
                arr = np.zeros((0, ), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(maccs_df['mols'][i]), arr)
                maccs_keys = np.vstack([maccs_keys, arr])
            except:
                maccs_keys = np.vstack([maccs_keys, np.full(BIT_LENGTH, np.nan)])

        for maccs_index in range(1, 167):
            maccs_df[f'maccs_{maccs_index}'] = maccs_keys[:, maccs_index]  #ignore index 0
        maccs_cols = [col for col in maccs_df.columns if 'maccs' in col]

        #merge with train, test
        train = train.merge(maccs_df, left_on='molecule_name', right_on='ids', how='left')
        test = test.merge(maccs_df, left_on='molecule_name', right_on='ids', how='left')

        self.train[maccs_cols] = train[maccs_cols]
        self.test[maccs_cols] = test[maccs_cols]


class AtomEnvironment(Feature):
    def create_features(self):
        global train, test

        #load in mol files
        with open('./data/input/mols_with_Hs.pickle', 'rb') as f:
            atom_env_df = pickle.load(f)

        train = train.merge(atom_env_df, left_on='molecule_name', right_on='ids')
        test = test.merge(atom_env_df, left_on='molecule_name', right_on='ids')

        train, cols = get_atom_env_feature(train)
        test, _ = get_atom_env_feature(test)

        for col in cols:
            self.train[col] = train[col]
            self.test[col] = test[col]


class AtomNeighbors(Feature):
    def create_features(self):
        global train, test

        #load in mol files 
        with open('./data/input/mols_with_Hs.pickle', 'rb') as f:
            neighbor_df = pickle.load(f)

        train = train.merge(neighbor_df, left_on='molecule_name', right_on='ids')
        test = test.merge(neighbor_df, left_on='molecule_name', right_on='ids')

        train, cols = get_atom_neighbor_feature(train)
        test, _ = get_atom_neighbor_feature(test)

        for col in cols:
            self.train[col] = train[col]
            self.test[col] = test[col]      



# https://www.kaggle.com/artgor/brute-force-feature-engineering
class BruteForce(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe('./data/input/structures.feather')

        def map_atom_info(df, atom_idx):
            df = pd.merge(df, structures, how='left', left_on=['molecule_name', f'atom_index_{atom_idx}'],
                          right_on=['molecule_name', 'atom_index'])
            df = df.drop('atom_index', axis=1)
            df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                    'x': f'x_{atom_idx}',
                                    'y': f'y_{atom_idx}',
                                    'z': f'z_{atom_idx}'})
            return df

        train = map_atom_info(train, 0)
        train = map_atom_info(train, 1)
        test = map_atom_info(test, 0)
        test = map_atom_info(test, 1)

        train_p_0 = train[['x_0', 'y_0', 'z_0']].values
        train_p_1 = train[['x_1', 'y_1', 'z_1']].values
        test_p_0 = test[['x_0', 'y_0', 'z_0']].values
        test_p_1 = test[['x_1', 'y_1', 'z_1']].values

        train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
        train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
        test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
        train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
        test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
        train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
        test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

        train['type_0'] = train['type'].apply(lambda x: x[0])
        test['type_0'] = test['type'].apply(lambda x: x[0])

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        original_cols = train.columns

        train = generate_brute_force_features(train)
        test = generate_brute_force_features(test)

        new_cols = [col for col in train.columns if col not in original_cols]

        for col in new_cols:
            self.train[col] = train[col]
            self.test[col] = test[col]      


# https://www.kaggle.com/hervind/speed-up-coulomb-interaction-56x-faster
class CoulombInteraction(Feature):
    def create_features(self):
        global train, test
        train.drop('scalar_coupling_constant', axis=1, inplace=True)
        total = pd.concat([train, test], axis=0)
        structures = feather.read_dataframe('./data/input/structures.feather')
        NCORES = 6
        NUM = 5

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)
        total = reduce_mem_usage(total)
        structures = reduce_mem_usage(structures)
        gc.collect()

        df_distance = structures.merge(structures, how = 'left', on= 'molecule_name', suffixes = ('_0', '_1'))
        # remove same molecule
        df_distance = df_distance.loc[df_distance['atom_index_0'] != df_distance['atom_index_1']]

        df_distance['distance'] = np.linalg.norm(df_distance[['x_0','y_0', 'z_0']].values - 
                                                df_distance[['x_1', 'y_1', 'z_1']].values, axis=1, ord = 2)

        def get_interaction_data_frame(df_distance, num_nearest = 5):
            print("START")
            
            # get nearest 5 (num_nearest) by distances
            df_temp = df_distance.groupby(['molecule_name', 'atom_index_0', 'atom_1'])['distance'].nsmallest(num_nearest)
            
            # make it clean
            df_temp = pd.DataFrame(df_temp).reset_index()[['molecule_name', 'atom_index_0', 'atom_1', 'distance']]
            df_temp.columns = ['molecule_name', 'atom_index', 'atom', 'distance']
            
            print("Time Nearest")
            
            # get rank by distance
            df_temp['distance_rank'] = df_temp.groupby(['molecule_name', 'atom_index', 'atom'])['distance'].rank(ascending = True, method = 'first').astype(int)
            
            print("Time Rank")
            
            # pivot to get nearest distance by atom type 
            df_distance_nearest = pd.pivot_table(df_temp, index = ['molecule_name','atom_index'], columns= ['atom', 'distance_rank'], values= 'distance')
            
            print("Time Pivot")
            del df_temp
            
            columns_distance_nearest =  np.core.defchararray.add('distance_nearest_', 
                                                np.array(df_distance_nearest.columns.get_level_values('distance_rank')).astype(str) +  
                                                np.array(df_distance_nearest.columns.get_level_values('atom')) )
            df_distance_nearest.columns = columns_distance_nearest
            
            # 1 / r^2 to get the square inverse same with the previous kernel
            df_distance_sq_inv_farthest = 1 / (df_distance_nearest ** 2)
            
            columns_distance_sq_inv_farthest = [col.replace('distance_nearest', 'dist_sq_inv_') for col in columns_distance_nearest]

            df_distance_sq_inv_farthest.columns = columns_distance_sq_inv_farthest
    
            print("Time Inverse Calculation")
            
            df_interaction = pd.concat([df_distance_sq_inv_farthest, df_distance_nearest] , axis = 1)
            df_interaction.reset_index(inplace = True)
      
            print("Time Concat")
            
            return df_interaction


        df_interaction = get_interaction_data_frame(df_distance)

        #merge with train, test
        train = train.merge(df_interaction, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
        train = train.merge(df_interaction, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

        test = test.merge(df_interaction, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
        test = test.merge(df_interaction, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

        train.fillna(0, inplace=True)
        test.fillna(0, inplace=True)

        col_name_list = [col for col in train.columns if 'sq_inv' in col]
        print(f'Number of cols generated is {len(col_name_list)}')
       
        for col in col_name_list:
            self.train[col] = train[col]
            self.test[col] = test[col]     


class CosBetweenClosest(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe("./data/input/structures.feather")
        df_distance = structures.merge(structures, on='molecule_name', how='left', suffixes=['_0', '_1'])
        df_distance = df_distance[df_distance['atom_index_0'] != df_distance['atom_index_1']]
        df_distance['distance'] = np.linalg.norm(df_distance[['x_0', 'y_0', 'z_0']].values - 
                                                df_distance[['x_1', 'y_1',  'z_1']].values, axis=1, ord = 2)

        df_distance['dist_a0'] = np.linalg.norm(df_distance[['x_0', 'y_0', 'z_0']].values, axis=1, ord = 2)
        df_distance['dist_a1'] = np.linalg.norm(df_distance[['x_1', 'y_1', 'z_1']].values, axis=1, ord = 2)
        df_distance['costheta'] = np.sum(np.multiply(df_distance[['x_0', 'y_0', 'z_0']].values, df_distance[['x_1', 'y_1', 'z_1']].values), axis=1) / (df_distance['dist_a0'] * df_distance['dist_a1'])

        tmp = df_distance.groupby(['molecule_name', 'atom_index_0'])['distance'].nsmallest(10)
        tmp = tmp.reset_index()
        df_distance = df_distance.reset_index()
        tmp = tmp.merge(df_distance, left_on=['molecule_name', 'atom_index_0', 'level_2'], right_on=['molecule_name', 'atom_index_0', 'index'], how='left')[['molecule_name', 'atom_index_0', 'costheta']]

        def convert_to_arr(df):
            len_cos_arr = 10
            len_prop_arr = 2
            cos_lis = []
            prop_lis = []
            groups = df.groupby(['molecule_name', 'atom_index_0'])
            for group in tqdm(groups):
                cur_num_cos = len(group[1]['costheta'])
                cos = np.pad(group[1]['costheta'].values, [0, len_cos_arr - cur_num_cos], 'constant')
                #zero pad when there are less than 10 dist values for the molecule&atom pair
                cos_lis.append(list(cos))
                prop_lis.append(list(group[0]))
                
            arr_combined = np.hstack([np.array(prop_lis), np.array(cos_lis)])
            col_names = [f'cos_between_closest_{i}' for i in range(1, 11)]
            col_names = ['molecule_name', 'atom_index_0'] + col_names
            return pd.DataFrame(arr_combined, columns=col_names)

        cos = convert_to_arr(tmp)
        cos['atom_index_0'] = cos['atom_index_0'].astype(int)

        train = train.merge(cos, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        train = train.merge(cos, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        test = test.merge(cos, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        test = test.merge(cos, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'closest' in col]
        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)  


class DistanceFromClosest(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe("./data/input/structures.feather")
        df_distance = structures.merge(structures, on='molecule_name', how='left', suffixes=['_0', '_1'])
        df_distance = df_distance[df_distance['atom_index_0'] != df_distance['atom_index_1']]
        df_distance['distance'] = np.linalg.norm(df_distance[['x_0', 'y_0', 'z_0']].values - 
                                                df_distance[['x_1', 'y_1',  'z_1']].values, axis=1, ord = 2)
        tmp = df_distance.groupby(['molecule_name', 'atom_index_0'])['distance'].nsmallest(10)
        tmp = tmp.reset_index()[['molecule_name', 'atom_index_0', 'distance']]

        def convert_to_arr(df):
            len_dist_arr = 10
            len_prop_arr = 2
            dist_lis = []
            prop_lis = []
            groups = df.groupby(['molecule_name', 'atom_index_0'])
            for group in tqdm(groups):
                cur_num_dist = len(group[1]['distance'])
                dist = np.pad(group[1]['distance'].values, [0, len_dist_arr - cur_num_dist], 'constant')
                #zero pad when there are less than 10 dist values for the molecule&atom pair
                dist_lis.append(list(dist))
                prop_lis.append(list(group[0]))
                
            arr_combined = np.hstack([np.array(prop_lis), np.array(dist_lis)])
            col_names = [f'dist_from_closest_{i}' for i in range(1, 11)]
            col_names = ['molecule_name', 'atom_index_0'] + col_names
            return pd.DataFrame(arr_combined, columns=col_names)

        dist = convert_to_arr(tmp)
        dist['atom_index_0'] = dist['atom_index_0'].astype(int)

        cols = [col for col in dist.columns if 'closest' in col]
        for col in cols:
            dist[col] = dist[col].astype(float)

        dist['dist_from_closest_mean'] = dist[cols].mean(axis=1)
        dist['dist_from_closest_max'] = dist[cols].max(axis=1)
        dist['dist_from_closest_min'] = dist[cols].min(axis=1)
        dist['dist_from_closest_std'] = dist[cols].std(axis=1) 
        dist['dist_from_closest_zerocount'] = (dist[cols] == 0.0).astype(int).sum(axis=1)       

        train = train.merge(dist, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        train = train.merge(dist, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        test = test.merge(dist, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        test = test.merge(dist, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'closest' in col]
        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)  


class ElectroNegFromClosest(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe("./data/input/structures.feather")
        df_distance = structures.merge(structures, on='molecule_name', how='left', suffixes=['_0', '_1'])
        df_distance = df_distance[df_distance['atom_index_0'] != df_distance['atom_index_1']]
        df_distance['distance'] = np.linalg.norm(df_distance[['x_0', 'y_0', 'z_0']].values - 
                                                df_distance[['x_1', 'y_1',  'z_1']].values, axis=1, ord = 2)
        tmp = df_distance.groupby(['molecule_name', 'atom_index_0'])['distance'].nsmallest(10)
        tmp = tmp.reset_index()
        df_distance = df_distance.reset_index()
        tmp = tmp.merge(df_distance, left_on=['molecule_name', 'atom_index_0', 'level_2'], right_on=['molecule_name', 'atom_index_0', 'index'], how='left')[['molecule_name', 'atom_index_0', 'atom_1']]

        en ={'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}
        tmp['en'] = tmp['atom_1'].map(en)

        def convert_to_arr(df):
            len_en_arr = 10
            len_prop_arr = 2
            en_lis = []
            prop_lis = []
            groups = df.groupby(['molecule_name', 'atom_index_0'])
            for group in tqdm(groups):
                cur_num_en = len(group[1]['en'])
                en = np.pad(group[1]['en'].values, [0, len_en_arr - cur_num_en], 'constant')
                #zero pad when there are less than 10 dist values for the molecule&atom pair
                en_lis.append(list(en))
                prop_lis.append(list(group[0]))
                
            arr_combined = np.hstack([np.array(prop_lis), np.array(en_lis)])
            col_names = [f'en_from_closest_{i}' for i in range(1, 11)]
            col_names = ['molecule_name', 'atom_index_0'] + col_names
            return pd.DataFrame(arr_combined, columns=col_names)

        en = convert_to_arr(tmp)
        en['atom_index_0'] = en['atom_index_0'].astype(int)

        cols = [col for col in en.columns if 'closest' in col]
        for col in cols:
            en[col] = en[col].astype(float)     

        en['en_from_closest_mean'] = en[cols].mean(axis=1)   
        en['en_from_closest_max'] = en[cols].max(axis=1)
        en['en_from_closest_min'] = en[cols].min(axis=1)
        en['en_from_closest_std'] = en[cols].std(axis=1)

        train = train.merge(en, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        train = train.merge(en, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        test = test.merge(en, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        test = test.merge(en, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'closest' in col]
        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)  


class ACSF(Feature):
    def create_features(self):
        #dscribeがWSLでしか動かないのでこの特徴を生成するときにimportする
        import dscribe
        from dscribe.descriptors import ACSF
        import ase

        global train, test
        structures = feather.read_dataframe("./data/input/structures.feather")
        groups = structures.groupby('molecule_name')

        acsf = ACSF(species=["H", "O", 'N', 'C', 'F'],
                    rcut=10.0,
                    g2_params=[[1, 2], [0.1, 2], [0.01, 2],
                                [1, 6], [0.1, 6], [0.01, 6]],
                    g4_params=[[1, 4,  1], [0.1, 4,  1], [0.01, 4,  1], 
                                [1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]],
                    )

        acsf_descs = []
        for group in tqdm(groups):
            atoms = group[1]['atom'].values
            positions = group[1][['x', 'y', 'z']].values
            atoms_obj = ase.Atoms(atoms, positions)
            acsf_desc = acsf.create(atoms_obj)
            for row in acsf_desc:
                acsf_descs.append(row)
        acsf_descs = pd.DataFrame(acsf_descs)   

        acsf_descs = acsf_descs.reset_index(drop=True)

        # drop all zero columns
        zero_cols = list(acsf_descs.columns[acsf_descs.sum(axis=0) < 5])
        print(f"Dropping {len(zero_cols)} almost all zero columns.")
        acsf_descs.drop(zero_cols, axis=1, inplace=True)

        acsf_descs.columns = [f'acsf_{i}' for i in range(acsf_descs.shape[1])]
        
        structures = pd.concat([structures, acsf_descs], axis=1)

        train = train.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
        train = train.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=['_a0', '_a1'])

        test = test.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
        test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'acsf' in col]
        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)  


class SOAP(Feature):
    def create_features(self):
        #dscribeがWSLでしか動かないのでこの特徴を生成するときにimportする
        import dscribe
        from dscribe.descriptors import SOAP
        import ase

        global train, test
        structures = feather.read_dataframe("./data/input/structures.feather")
        groups = structures.groupby('molecule_name')

        soap = SOAP(species=["H", "O", 'N', 'C', 'F'],
                    rcut=10.0,
                    nmax=3,
                    lmax=3
                    )

        soap_descs = []
        for group in tqdm(groups):
            atoms = group[1]['atom'].values
            positions = group[1][['x', 'y', 'z']].values
            atoms_obj = ase.Atoms(atoms, positions)
            soap_desc = soap.create(atoms_obj)
            for row in soap_desc:
                soap_descs.append(row)
        soap_descs = pd.DataFrame(soap_descs)   

        soap_descs = soap_descs.reset_index(drop=True)

        # drop all zero columns
        zero_cols = list(soap_descs.columns[soap_descs.sum(axis=0) < 5])
        print(f"Dropping {len(zero_cols)} almost all zero columns.")
        soap_descs.drop(zero_cols, axis=1, inplace=True)

        soap_descs.columns = [f'soap_{i}' for i in range(soap_descs.shape[1])]
        
        structures = pd.concat([structures, soap_descs], axis=1)

        train = train.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
        train = train.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=['_a0', '_a1'])

        test = test.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
        test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'soap' in col]
        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)  


# https://www.kaggle.com/scirpus/angles-and-distances
class OpenBabelBasic(Feature):
    def create_features(self):
        import openbabel

        global train, test

        #train = feather.read_dataframe('./data/iuput/train.feather')
        #test = feather.read_dataframe('./data/iuput/test.feather')
        structures = feather.read_dataframe('./data/input/structures.feather')

        # get number of total atoms in a molecule
        tmp = structures.groupby('molecule_name')['atom_index'].max().reset_index(drop=False)
        tmp.columns = ['molecule_name', 'totalatoms']
        tmp['totalatoms'] +=1
        train = train.merge(tmp, on='molecule_name')
        test = test.merge(tmp, on='molecule_name')

        # convert xyz to OBMol
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("xyz")
        structdir='./data/input/structures/'
        mols=[]
        mols_files=os.listdir(structdir)
        mols_index=dict(map(reversed, enumerate(mols_files)))
        for f in tqdm(mols_index.keys()):
            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, structdir+f) 
            mols.append(mol)

        # calculate basic stats using openbabel
        def calc_obabel(df):
            stats = []
            for m,groupdf in tqdm(df.groupby('molecule_name')):
                mol=mols[mols_index[m+'.xyz']]
                for i in groupdf.index.values:
                    totalatoms = groupdf.loc[i].totalatoms
                    firstatomid = int(groupdf.loc[i].atom_index_0)
                    secondatomid = int(groupdf.loc[i].atom_index_1)
                    entrystats = {}
                    entrystats['totalatoms'] = totalatoms
                    #entrystats['scalar_coupling_constant'] = float(groupdf.loc[i].scalar_coupling_constant)
                    #entrystats['type'] = groupdf.loc[i]['type']
                    a = mol.GetAtomById(firstatomid)
                    b = mol.GetAtomById(secondatomid)
                    #entrystats['molecule_name'] = m
                    #entrystats['atom_index_0'] = firstatomid
                    #entrystats['atom_index_1'] = secondatomid
                    #entrystats['bond_distance'] = a.GetDistance(b)
                    entrystats['bond_atom'] = b.GetType()

                    #Put the tertiary data in order of distance from first hydrogen
                    tertiarystats = {}
                    for j,c in enumerate(list(set(range(totalatoms)).difference(set([firstatomid,secondatomid])))):
                        tertiaryatom = mol.GetAtomById(c)
                        tp = tertiaryatom.GetType()
                        dist = a.GetDistance(tertiaryatom)
                        ang = a.GetAngle(b,tertiaryatom)*math.pi/180
                        while(dist in tertiarystats):
                            dist += 1e-15
                            # print('Duplicates!',m,j,dist)
                        tertiarystats[dist] = [tp,dist,ang]
                    
                    for k, c in enumerate(sorted(tertiarystats.keys())):
                        entrystats['tertiary_atom_'+str(k)] = tertiarystats[c][0]
                        entrystats['tertiary_distance_'+str(k)] = tertiarystats[c][1]
                        entrystats['tertiary_angle_'+str(k)] = tertiarystats[c][2]

                    # to ensure every atom pair has the same length
                    if len(tertiarystats) < 10:
                        for l in range(len(tertiarystats), 10):
                            entrystats['tertiary_atom_'+str(l)] = np.nan
                            entrystats['tertiary_distance_'+str(l)] = np.nan
                            entrystats['tertiary_angle_'+str(l)] = np.nan

                    if len(tertiarystats) > 10:
                        for m in range(10, len(tertiarystats)):
                            del entrystats['tertiary_atom_'+str(m)]
                            del entrystats['tertiary_distance_'+str(m)]
                            del entrystats['tertiary_angle_'+str(m)]

                    stats.append(entrystats)

            return pd.DataFrame(stats)

        print('Starting train')
        ob_train = calc_obabel(train)
        print('Starting test')
        ob_test = calc_obabel(test)

        cols = ['bond_atom', 'totalatoms']
        for n in range(10):
            new = ['tertiary_atom_'+str(n), 'tertiary_distance_'+str(n), 'tertiary_angle_'+str(n)]
            cols += new

        self.train[cols] = ob_train[cols]
        self.test[cols] = ob_test[cols]


class DistanceByAtom(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe('./data/input/structures.feather')
        dist = structures.merge(structures, on=['molecule_name'], how='left', suffixes=['_0', '_1'])
        dist['distance'] = np.linalg.norm(dist[['x_0', 'y_0', 'z_0']].values - 
                                                        dist[['x_1', 'y_1',  'z_1']].values, axis=1, ord = 2)
        dist = dist[dist['atom_index_0'] != dist['atom_index_1']]

        def convert_to_arr(df, atom, nsmallest):
            print(f'Starting for {atom}')
            len_dist_arr = nsmallest
            len_prop_arr = 2
            dist_lis = []
            prop_lis = []
            groups = df.groupby(['molecule_name', 'atom_index_0'])
            for group in tqdm(groups):
                cur_num_dist = len(group[1]['distance'])
                dist = np.pad(group[1]['distance'].values, [0, len_dist_arr - cur_num_dist], 'constant')
                #zero pad when there are less than 10 dist values for the molecule&atom pair
                dist_lis.append(list(dist))
                prop_lis.append(list(group[0]))

            arr_combined = np.hstack([np.array(prop_lis), np.array(dist_lis)])
            col_names = [f'dist_by_atom_{atom}_{i}' for i in range(1, nsmallest+1)]
            col_names = ['molecule_name', 'atom_index_0'] + col_names
            return pd.DataFrame(arr_combined, columns=col_names)

        def calc_dist_by_atom(dist, atom, nsmallest):
            # get the dist for the target atom
            dist = dist[dist['atom_1'] == atom]
            tmp = dist.groupby(['molecule_name', 'atom_index_0'])['distance'].nsmallest(nsmallest)
            tmp = tmp.reset_index()[['molecule_name', 'atom_index_0', 'distance']]
            
            dist = convert_to_arr(tmp, atom, nsmallest)
            dist['atom_index_0'] = dist['atom_index_0'].astype(int)
            cols = [col for col in dist.columns if 'by_atom' in col]
            for col in cols:
                dist[col] = dist[col].astype(float)
                
            dist[f'dist_by_atom_{atom}_mean'] = dist[cols].mean(axis=1)
            dist[f'dist_by_atom_{atom}_max'] = dist[cols].max(axis=1)
            dist[f'dist_by_atom_{atom}_min'] = dist[cols].min(axis=1)
            dist[f'dist_by_atom_{atom}_std'] = dist[cols].std(axis=1) 
            dist[f'dist_by_atom_{atom}_zerocount'] = (dist[cols] == 0.0).astype(int).sum(axis=1)       

            return dist

        dist_h = calc_dist_by_atom(dist, 'H', 5)
        dist_c = calc_dist_by_atom(dist, 'C', 5).drop(['molecule_name', 'atom_index_0'], axis=1)
        dist_o = calc_dist_by_atom(dist, 'O', 5).drop(['molecule_name', 'atom_index_0'], axis=1)
        dist_n = calc_dist_by_atom(dist, 'N', 5).drop(['molecule_name', 'atom_index_0'], axis=1)
        dist_f = calc_dist_by_atom(dist, 'F', 5).drop(['molecule_name', 'atom_index_0'], axis=1)

        dist = pd.concat([dist_h, dist_c, dist_o, dist_n, dist_f], axis=1)
        print(f'Columns of dist are {dist.columns.values}')

        train = train.merge(dist, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        train = train.merge(dist, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])
        test = test.merge(dist, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        test = test.merge(dist, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'by_atom' in col]
        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)  
       

# https://www.kaggle.com/todnewman/keras-neural-net-for-champs
class CosineDistance(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe('./data/input/structures.feather')

        structures['c_x'] = structures.groupby('molecule_name')['x'].transform('mean')
        structures['c_y'] = structures.groupby('molecule_name')['y'].transform('mean')
        structures['c_z'] = structures.groupby('molecule_name')['z'].transform('mean')

        def map_atom_info(df, atom_idx):
            df = pd.merge(df, structures, how='left', left_on=['molecule_name', f'atom_index_{atom_idx}'],
                          right_on=['molecule_name', 'atom_index'])
            df = df.drop('atom_index', axis=1)
            df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                    'x': f'x_{atom_idx}',
                                    'y': f'y_{atom_idx}',
                                    'z': f'z_{atom_idx}',
                                    'c_x': f'c_x_{atom_idx}',
                                    'c_y': f'c_y_{atom_idx}',
                                    'c_z': f'c_z_{atom_idx}'})
            return df

        train = map_atom_info(train, 0)
        train = map_atom_info(train, 1)
        test = map_atom_info(test, 0)
        test = map_atom_info(test, 1)       

        train_p_0 = train[['x_0', 'y_0', 'z_0']].values
        train_p_1 = train[['x_1', 'y_1', 'z_1']].values
        test_p_0 = test[['x_0', 'y_0', 'z_0']].values
        test_p_1 = test[['x_1', 'y_1', 'z_1']].values

        train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)


        structures = feather.read_dataframe('./data/input/structures.feather')
        temp = structures.merge(structures, on="molecule_name", how="left", suffixes=["_0", "_1"])
        temp = temp[temp["atom_index_0"] != temp["atom_index_1"]]
        temp_p_0 = temp[['x_0', 'y_0', 'z_0']].values
        temp_p_1 = temp[['x_1', 'y_1', 'z_1']].values
        temp['dist'] = np.linalg.norm(temp_p_0 - temp_p_1, axis=1)

        temp_min = temp.copy()
        temp_min["min_dist"] = temp.groupby(["molecule_name", "atom_index_0"])["dist"].transform('min')
        temp_min = temp_min[temp_min["dist"] == temp_min["min_dist"]].groupby(["molecule_name", "atom_index_0"]).first().reset_index()
        drop = ["atom_index_1","atom_0", "x_0", "y_0", "z_0", "atom_1", "dist", "min_dist"]
        temp_min.drop(drop, axis=1, inplace=True)
        temp_min.columns = ["molecule_name", "atom_index_min", "x_closest", "y_closest", "z_closest"]

        temp_max = temp.copy()
        temp_max["max_dist"] = temp.groupby(["molecule_name", "atom_index_0"])["dist"].transform('max')
        temp_max = temp_max[temp_max["dist"] == temp_max["max_dist"]].groupby(["molecule_name", "atom_index_0"]).first().reset_index()
        drop = ["atom_index_1","atom_0", "x_0", "y_0", "z_0", "atom_1", "dist", "max_dist"]
        temp_max.drop(drop, axis=1, inplace=True)
        temp_max.columns = ["molecule_name", "atom_index_max", "x_furthest", "y_furthest", "z_furthest"]

        train = train.merge(temp_min, left_on=["molecule_name", "atom_index_0"], right_on=["molecule_name", "atom_index_min"], how="left")
        train = train.merge(temp_min, left_on=["molecule_name","atom_index_1"], right_on=["molecule_name","atom_index_min"], how="left", suffixes=['_0', "_1"])
        test = test.merge(temp_min, left_on=["molecule_name", "atom_index_0"], right_on=["molecule_name", "atom_index_min"], how="left")
        test = test.merge(temp_min, left_on=["molecule_name","atom_index_1"], right_on=["molecule_name","atom_index_min"], how="left", suffixes=['_0', "_1"])

        train = train.merge(temp_max, left_on=["molecule_name", "atom_index_0"], right_on=["molecule_name", "atom_index_max"], how="left")
        train = train.merge(temp_max, left_on=["molecule_name","atom_index_1"], right_on=["molecule_name","atom_index_max"], how="left", suffixes=['_0', "_1"])
        test = test.merge(temp_max, left_on=["molecule_name", "atom_index_0"], right_on=["molecule_name", "atom_index_max"], how="left")
        test = test.merge(temp_max, left_on=["molecule_name","atom_index_1"], right_on=["molecule_name","atom_index_max"], how="left", suffixes=['_0', "_1"])


        def add_features(df):
            #df = df.merge(structures[["molecule_name", "c_x", "c_y", "c_z"]], on="molecule_name", how="left")

            df["distance_center0"]=((df['x_0']-df['c_x_0'])**2+(df['y_0']-df['c_y_0'])**2+(df['z_0']-df['c_z_0'])**2)**(1/2)
            df["distance_center1"]=((df['x_1']-df['c_x_1'])**2+(df['y_1']-df['c_y_1'])**2+(df['z_1']-df['c_z_1'])**2)**(1/2)
            df["distance_c0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)
            df["distance_c1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)
            df["distance_f0"]=((df['x_0']-df['x_furthest_0'])**2+(df['y_0']-df['y_furthest_0'])**2+(df['z_0']-df['z_furthest_0'])**2)**(1/2)
            df["distance_f1"]=((df['x_1']-df['x_furthest_1'])**2+(df['y_1']-df['y_furthest_1'])**2+(df['z_1']-df['z_furthest_1'])**2)**(1/2)
            df["vec_center0_x"]=(df['x_0']-df['c_x_0'])/(df["distance_center0"]+1e-10)
            df["vec_center0_y"]=(df['y_0']-df['c_y_0'])/(df["distance_center0"]+1e-10)
            df["vec_center0_z"]=(df['z_0']-df['c_z_0'])/(df["distance_center0"]+1e-10)
            df["vec_center1_x"]=(df['x_1']-df['c_x_1'])/(df["distance_center1"]+1e-10)
            df["vec_center1_y"]=(df['y_1']-df['c_y_1'])/(df["distance_center1"]+1e-10)
            df["vec_center1_z"]=(df['z_1']-df['c_z_1'])/(df["distance_center1"]+1e-10)
            df["vec_c0_x"]=(df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)
            df["vec_c0_y"]=(df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)
            df["vec_c0_z"]=(df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)
            df["vec_c1_x"]=(df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)
            df["vec_c1_y"]=(df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)
            df["vec_c1_z"]=(df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)
            df["vec_f0_x"]=(df['x_0']-df['x_furthest_0'])/(df["distance_f0"]+1e-10)
            df["vec_f0_y"]=(df['y_0']-df['y_furthest_0'])/(df["distance_f0"]+1e-10)
            df["vec_f0_z"]=(df['z_0']-df['z_furthest_0'])/(df["distance_f0"]+1e-10)
            df["vec_f1_x"]=(df['x_1']-df['x_furthest_1'])/(df["distance_f1"]+1e-10)
            df["vec_f1_y"]=(df['y_1']-df['y_furthest_1'])/(df["distance_f1"]+1e-10)
            df["vec_f1_z"]=(df['z_1']-df['z_furthest_1'])/(df["distance_f1"]+1e-10)
            df["vec_x"]=(df['x_1']-df['x_0'])/df["dist"]
            df["vec_y"]=(df['y_1']-df['y_0'])/df["dist"]
            df["vec_z"]=(df['z_1']-df['z_0'])/df["dist"]
            df["cos_c0_c1"]=df["vec_c0_x"]*df["vec_c1_x"]+df["vec_c0_y"]*df["vec_c1_y"]+df["vec_c0_z"]*df["vec_c1_z"]
            df["cos_f0_f1"]=df["vec_f0_x"]*df["vec_f1_x"]+df["vec_f0_y"]*df["vec_f1_y"]+df["vec_f0_z"]*df["vec_f1_z"]
            df["cos_center0_center1"]=df["vec_center0_x"]*df["vec_center1_x"]+df["vec_center0_y"]*df["vec_center1_y"]+df["vec_center0_z"]*df["vec_center1_z"]
            df["cos_c0"]=df["vec_c0_x"]*df["vec_x"]+df["vec_c0_y"]*df["vec_y"]+df["vec_c0_z"]*df["vec_z"]
            df["cos_c1"]=df["vec_c1_x"]*df["vec_x"]+df["vec_c1_y"]*df["vec_y"]+df["vec_c1_z"]*df["vec_z"]
            df["cos_f0"]=df["vec_f0_x"]*df["vec_x"]+df["vec_f0_y"]*df["vec_y"]+df["vec_f0_z"]*df["vec_z"]
            df["cos_f1"]=df["vec_f1_x"]*df["vec_x"]+df["vec_f1_y"]*df["vec_y"]+df["vec_f1_z"]*df["vec_z"]
            df["cos_center0"]=df["vec_center0_x"]*df["vec_x"]+df["vec_center0_y"]*df["vec_y"]+df["vec_center0_z"]*df["vec_z"]
            df["cos_center1"]=df["vec_center1_x"]*df["vec_x"]+df["vec_center1_y"]*df["vec_y"]+df["vec_center1_z"]*df["vec_z"]
            df=df.drop(['vec_c0_x','vec_c0_y','vec_c0_z','vec_c1_x','vec_c1_y','vec_c1_z',
                        'vec_f0_x','vec_f0_y','vec_f0_z','vec_f1_x','vec_f1_y','vec_f1_z',
                        'vec_center0_x','vec_center0_y','vec_center0_z','vec_center1_x','vec_center1_y','vec_center1_z',
                        'vec_x','vec_y','vec_z'], axis=1)
            return df
                    
        train = add_features(train)
        test = add_features(test)

        new_cols = ['c_x_0',
                    'c_y_0', 'c_z_0', 'c_x_1', 'c_y_1',
                    'c_z_1', 'x_closest_0', 'y_closest_0',
                    'z_closest_0','x_closest_1', 'y_closest_1',
                    'z_closest_1', 'x_furthest_0', 'y_furthest_0',
                    'z_furthest_0','x_furthest_1', 'y_furthest_1',
                    'z_furthest_1', 'distance_center0', 'distance_center1',
                    'distance_c0', 'distance_c1', 'distance_f0', 'distance_f1',
                    'cos_c0_c1', 'cos_f0_f1', 'cos_center0_center1', 'cos_c0',
                    'cos_c1', 'cos_f0', 'cos_f1', 'cos_center0', 'cos_center1']

        for col in new_cols:
            self.train[col] = train[col].astype(float)
            self.test[col] = test[col].astype(float)          


# https://www.kaggle.com/pridegoodmusic/cis-trans-isomerism-feature
class CisTrans(Feature):
    def create_features(self):
        global train, test
        structures = feather.read_dataframe('./data/input/structures.feather')

        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("xyz")

        def cis_trans_bond_indices(molecule_name):
            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, f'./data/input/structures/{molecule_name}.xyz')
            obs = openbabel.OBStereoFacade(mol)
            has_ct = [obs.HasCisTransStereo(n) for n in range(mol.NumBonds())]
            return [i for i, x in enumerate(has_ct) if x == True] if has_ct else []

        df = pd.DataFrame(structures.molecule_name.unique(), columns=['molecule_name'])
        df['bond_indices'] = df["molecule_name"].progress_apply(lambda x: cis_trans_bond_indices(x))
        df['len_bond_indices'] = df["bond_indices"].progress_apply(lambda x:len(x))

        train = pd.merge(train, df, how='left', on='molecule_name')
        test = pd.merge(test, df, how='left', on='molecule_name')

        def is_cis_trans(molecule_name, bond_indices, atom_index_0, atom_index_1):
            if len(bond_indices) == 0:
                return pd.Series([0,0])

            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, f'./data/input/structures/{molecule_name}.xyz')
            obs = openbabel.OBStereoFacade(mol)
            
            is_cis   = [obs.GetCisTransStereo(i).IsCis(atom_index_0, atom_index_1) for i in bond_indices]
            is_trans = [obs.GetCisTransStereo(i).IsTrans(atom_index_0, atom_index_1) for i in bond_indices]
            return pd.Series([int(True in is_cis), int(True in is_trans)])      

        train[['is_cis','is_trans']] = train.progress_apply(lambda x: is_cis_trans(x.molecule_name,
                                                                                    x.bond_indices,
                                                                                    x.atom_index_0,
                                                                                    x.atom_index_1), axis=1)

        test[['is_cis','is_trans']] = test.progress_apply(lambda x: is_cis_trans(x.molecule_name,
                                                                                    x.bond_indices,
                                                                                    x.atom_index_0,
                                                                                    x.atom_index_1), axis=1)  
        # https://www.kaggle.com/soerendip/angle-and-dihedral-for-the-champs-structures
        angles = pd.read_csv('./data/input/angles.csv')

        train = pd.merge(train, 
                        angles[['molecule_name','atom_index_0','atom_index_1','dihedral', 'cosinus']],
                        how='left',
                        on=['molecule_name','atom_index_0','atom_index_1'])

        test = pd.merge(test, 
                        angles[['molecule_name','atom_index_0','atom_index_1','dihedral', 'cosinus']],
                        how='left',
                        on=['molecule_name','atom_index_0','atom_index_1'])

        new_cols = ["is_cis", "is_trans", "dihedral", "cosinus"]

        for col in new_cols:
            self.train[col] = train[col]
            self.test[col] = test[col]      

if __name__ == '__main__':
    args = get_arguments()

    train = feather.read_dataframe('./data/input/train.feather')
    test = feather.read_dataframe('./data/input/test.feather')

    generate_features(globals(), args.force, args.which)
