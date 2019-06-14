import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromMolBlock

parser = ArgumentParser()
parser.add_argument('--removeHs', '-k', action='store_true', help='remove H atoms when generating mol')


if __name__ == '__main__':
    args = parser.parse_args()

    ids = []
    mols = []
    for i in range(1, 133886):
        ids.append(f'dsgdb9nsd_{i:06}')
        try:
            with open(f'./data/input/structures/dsgdb9nsd_{i:06}.mol', 'r') as mol:
                mols.append(MolFromMolBlock(mol.read(), removeHs=args.removeHs))
        except:
            mols.append(np.nan)

    df = pd.DataFrame()
    df['ids'] = ids
    df['mols'] = mols 

    if args.removeHs:
        with open('./data/input/mols_without_Hs.pickle', 'wb') as f:
            pickle.dump(df, f)
    else:
        with open('./data/input/mols_with_Hs.pickle', 'wb') as f:
            pickle.dump(df, f)       