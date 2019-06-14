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

from features.base import get_arguments, get_features, generate_features, Feature
from utils import get_atom_env_feature, get_atom_neighbor_feature, calc_atom_neighbor_feature, reduce_mem_usage

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

        self.train['x_0'] = train['x_0']
        self.train['x_1'] = train['x_1']
        self.train['y_0'] = train['y_0']
        self.train['y_1'] = train['y_1']
        self.train['z_0'] = train['z_0']
        self.train['z_1'] = train['z_1']        
        self.test['x_0'] = test['x_0']
        self.test['x_1'] = test['x_1']
        self.test['y_0'] = test['y_0']
        self.test['y_1'] = test['y_1']
        self.test['z_0'] = test['z_0']
        self.test['z_1'] = test['z_1']     


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


#https://www.kaggle.com/brandenkmurray/coulomb-interaction-parallelized
class CoulombInteraction(Feature):
    def create_features(self):
        global train, test
        train.drop('scalar_coupling_constant', axis=1, inplace=True)
        total = pd.concat([train, test], axis=0)
        structures = feather.read_dataframe('./data/input/structures.feather')
        NCORES = 2
        NUM = 5

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)
        total = reduce_mem_usage(total)
        structures = reduce_mem_usage(structures)
        gc.collect()

        def get_dist_matrix(structures, molecule):
            df_temp = structures.query('molecule_name == "{}"'.format(molecule))
            locs = df_temp[['x','y','z']].values
            num_atoms = len(locs)
            loc_tile = np.tile(locs.T, (num_atoms,1,1))
            dist_mat = ((loc_tile - loc_tile.T)**2).sum(axis=1)
            return dist_mat

        def assign_atoms_index(df, molecule):
            se_0 = df.query('molecule_name == "{}"'.format(molecule))['atom_index_0']
            se_1 = df.query('molecule_name == "{}"'.format(molecule))['atom_index_1']
            assign_idx = pd.concat([se_0, se_1]).unique()
            assign_idx.sort()
            return assign_idx

        def get_pickup_dist_matrix(df, structures, molecule, num_pickup=NUM, atoms=['H', 'C', 'N', 'O', 'F']):
            pickup_dist_matrix = np.zeros([0, len(atoms)*num_pickup])
            assigned_idxs = assign_atoms_index(df, molecule) # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]
            dist_mat = get_dist_matrix(structures, molecule)
            for idx in assigned_idxs: # [1, 2, 3, 4, 5, 6] -> [2]
                dist_arr = dist_mat[idx] # (7, 7) -> (7, )

                atoms_mole = structures.query('molecule_name == "{}"'.format(molecule))['atom'].values # ['O', 'C', 'C', 'N', 'H', 'H', 'H']
                atoms_mole_idx = structures.query('molecule_name == "{}"'.format(molecule))['atom_index'].values # [0, 1, 2, 3, 4, 5, 6]

                mask_atoms_mole_idx = atoms_mole_idx != idx # [ True,  True, False,  True,  True,  True,  True]
                masked_atoms = atoms_mole[mask_atoms_mole_idx] # ['O', 'C', 'N', 'H', 'H', 'H']
                masked_atoms_idx = atoms_mole_idx[mask_atoms_mole_idx]  # [0, 1, 3, 4, 5, 6]
                masked_dist_arr = dist_arr[mask_atoms_mole_idx]  # [ 5.48387003, 2.15181049, 1.33269675, 10.0578779, 4.34733927, 4.34727838]

                sorting_idx = np.argsort(masked_dist_arr) # [2, 1, 5, 4, 0, 3]
                sorted_atoms_idx = masked_atoms_idx[sorting_idx] # [3, 1, 6, 5, 0, 4]
                sorted_atoms = masked_atoms[sorting_idx] # ['N', 'C', 'H', 'H', 'O', 'H']
                sorted_dist_arr = 1 / masked_dist_arr[sorting_idx] #[0.75035825,0.46472494,0.23002898,0.23002576,0.18235297,0.09942455]

                target_matrix = np.zeros([len(atoms), num_pickup])
                for a, atom in enumerate(atoms):
                    pickup_atom = sorted_atoms == atom # [False, False,  True,  True, False,  True]
                    pickup_dist = sorted_dist_arr[pickup_atom] # [0.23002898, 0.23002576, 0.09942455]
                    num_atom = len(pickup_dist)
                    if num_atom > num_pickup:
                        target_matrix[a, :] = pickup_dist[:num_pickup]
                    else:
                        target_matrix[a, :num_atom] = pickup_dist
                pickup_dist_matrix = np.vstack([pickup_dist_matrix, target_matrix.reshape(-1)])
            return pickup_dist_matrix

        mols = total['molecule_name'].unique()
        dist_mat = np.zeros([0, NUM*5])
        atoms_idx = np.zeros([0], dtype=np.int32)
        molecule_names = np.empty([0])

        for mol in tqdm(mols):
            assigned_idxs = assign_atoms_index(total, mol)
            dist_mat_mole = get_pickup_dist_matrix(total, structures, mol, num_pickup=NUM)
            mol_name_arr = [mol] * len(assigned_idxs) 
            
            molecule_names = np.hstack([molecule_names, mol_name_arr])
            atoms_idx = np.hstack([atoms_idx, assigned_idxs])
            dist_mat = np.vstack([dist_mat, dist_mat_mole])

        col_name_list = []
        atoms = ['H', 'C', 'N', 'O', 'F']
        for a in atoms:
            for n in range(NUM):
                col_name_list.append('dist_{}_{}'.format(a, n))
                
        se_mole = pd.Series(molecule_names, name='molecule_name')
        se_atom_idx = pd.Series(atoms_idx, name='atom_index')
        df_dist = pd.DataFrame(dist_mat, columns=col_name_list)
        df_distance = pd.concat([se_mole, se_atom_idx,df_dist], axis=1)

        #merge with train, test
        train = train.merge(df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
        train = train.merge(df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

        test = test.merge(df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
        test = test.merge(df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

        for col in col_name_list:
            self.train[col] = train[col]
            self.test[col] = test[col]           




if __name__ == '__main__':
    args = get_arguments()

    train = feather.read_dataframe('./data/input/train.feather')
    test = feather.read_dataframe('./data/input/test.feather')

    generate_features(globals(), args.force, args.which)
