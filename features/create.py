import sys
sys.path.append('.')
import re

import pandas as pd
import numpy as np
import feather
from rdkit.Chem import Descriptors, Descriptors3D, MolFromMolBlock, MACCSkeys, DataStructs

from features.base import get_arguments, get_features, generate_features, Feature
from utils import get_atom_env_feature

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

        #calculate descriptors for each molecule from mol file
        ids = []
        mols = []
        for i in range(1, 133886):
            ids.append(f'dsgdb9nsd_{i:06}')
            try:
                with open(f'./data/input/structures/dsgdb9nsd_{i:06}.mol', 'r') as mol:
                    mols.append(MolFromMolBlock(mol.read()))
            except:
                mols.append(np.nan)
        rdkit_desc_df = pd.DataFrame()
        rdkit_desc_df['ids'] = ids
        rdkit_desc_df['mols'] = mols

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

        #calculate maccskeys for each molecule from mol file
        ids = []
        mols = []
        for i in range(1, 133886):
            ids.append(f'dsgdb9nsd_{i:06}')
            try:
                with open(f'./data/input/structures/dsgdb9nsd_{i:06}.mol', 'r') as mol:
                    mols.append(MolFromMolBlock(mol.read()))
            except:
                mols.append(np.nan)
        maccs_df = pd.DataFrame()
        maccs_df['ids'] = ids
        maccs_df['mols'] = mols 

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

        ids = []
        mols = []
        for i in range(1, 133886):
            ids.append(f'dsgdb9nsd_{i:06}')
            try:
                with open(f'./data/input/structures/dsgdb9nsd_{i:06}.mol', 'r') as mol:
                    mols.append(MolFromMolBlock(mol.read(), removeHs=False))
            except:
                mols.append(np.nan)
        neighbor_df = pd.DataFrame()
        neighbor_df['ids'] = ids
        neighbor_df['mols'] = mols 

        train = train.merge(neighbor_df, left_on='molecule_name', right_on='ids')
        test = test.merge(neighbor_df, left_on='molecule_name', right_on='ids')

        train, cols = get_atom_env_feature(train)
        test, _ = get_atom_env_feature(test)

        for col in cols:
            self.train[col] = train[col]
            self.test[col] = test[col]






if __name__ == '__main__':
    args = get_arguments()

    train = feather.read_dataframe('./data/input/train.feather')
    test = feather.read_dataframe('./data/input/test.feather')

    generate_features(globals(), args.force)
