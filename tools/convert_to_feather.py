import argparse
import pandas as pd
from pathlib import Path

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='File path to convert to feather format')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    stem = Path(args.file).stem
    par_dir = Path(args.file).parent

    df = pd.read_csv(str(par_dir) + '/' + str(stem) + '.csv')
    df.to_feather(str(par_dir) + '/' + str(stem) + '.feather')
