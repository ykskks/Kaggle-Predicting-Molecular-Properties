import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import feather
from sklearn.preprocessing import LabelEncoder

from features.base import get_arguments, get_features, generate_features, Feature

Feature.base_dir = 'features'


class CouplingType(Feature):
    def create_features(self):
        le = LabelEncoder()
        self.train['type'] = le.fit_transform(train['type'])
        self.test['type'] = le.transform(test['type'])


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

        le = LabelEncoder()
        self.train['atom_0'] = le.fit_transform(train['atom_0'])
        self.train['atom_1'] = le.fit_transform(train['atom_1'])
        self.test['atom_0'] = le.transform(test['atom_0'])
        self.test['atom_1'] = le.transform(test['atom_1'])


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


if __name__ == '__main__':
    args = get_arguments()

    train = feather.read_dataframe('./data/input/train.feather')
    test = feather.read_dataframe('./data/input/test.feather')

    generate_features(globals(), args.force)
