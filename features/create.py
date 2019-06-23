import sys
sys.path.append('.')
import re
import gc
import pickle

import pandas as pd
import numpy as np
import feather
from rdkit.Chem import Descriptors, Descriptors3D, MolFromMolBlock, MACCSkeys, DataStructs
from tqdm import tqdm
from joblib import Parallel, delayed

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

        self.train['atom_distance'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        self.test['atom_distance'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)


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

        col_name_list = [col for col in train.columns if 'sq_inv' in col]
        print(f'Number of cols generated is {len(col_name_list)}')
       
        for col in col_name_list:
            self.train[col] = train[col]
            self.test[col] = test[col]     


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
            dist_arr = np.zeros(len_dist_arr)
            prop_arr = np.zeros(len_prop_arr)
            groups = df.groupby(['molecule_name', 'atom_index_0'])
            for group in tqdm(groups):
                cur_num_dist = len(group[1]['distance'])
                dist = np.pad(group[1]['distance'].values, [0, len_dist_arr - cur_num_dist], 'constant')
                #zero pad when there are less than 10 dist values for the molecule&atom pair
                dist_arr = np.vstack([dist_arr, dist])
                prop_arr = np.vstack([prop_arr, np.array(group[0])])
                
            arr_combined = np.hstack([prop_arr[1:], dist_arr[1:]])
            col_names = [f'dist_from_closest_{i}' for i in range(1, 11)]
            col_names = ['molecule_name', 'atom_index_0'] + col_names
            return pd.DataFrame(arr_combined, columns=col_names)

        dist = convert_to_arr(tmp)
        dist['atom_index_0'] = dist['atom_index_0'].astype(int)

        train = train.merge(dist, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        train = train.merge(dist, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        test = test.merge(dist, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index_0'], how='left')
        test = test.merge(dist, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0'], how='left', suffixes=['_a0', '_a1'])

        new_cols = [col for col in train.columns if 'closest' in col]
        for col in new_cols:
            self.train[col] = train[col]
            self.test[col] = test[col]     

if __name__ == '__main__':
    args = get_arguments()

    train = feather.read_dataframe('./data/input/train.feather')
    test = feather.read_dataframe('./data/input/test.feather')

    generate_features(globals(), args.force, args.which)
