import argparse
import json
import logging
import datetime
import gc

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np
import pandas as pd
import feather

from utils import load_datasets, load_target, get_categorical_feats, calculate_metric, log_evaluation, reduce_mem_usage

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration for this run")
parser.add_argument("--keep_nans", action="store_true", help="keep nan values as they are")
parser.add_argument("--debug", action="store_true", help="activate debug mode")
args = parser.parse_args()
config_path = args.config
is_debug_mode = args.debug
keep_nans = args.keep_nans


#get config fot this run
with open(config_path, 'r') as f:
    config = json.load(f)


#prepare the log file
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
sc = logging.StreamHandler()
logger.addHandler(sc)
fh = logging.FileHandler(f'logs/log_{now}.log')
logger.addHandler(fh)
logger.debug(f'logs/log_{now}.log')
logger.debug(config_path)
logger.debug(f'is_debug_mode: {is_debug_mode}')
logger.debug(f'keep_nans: {keep_nans}')


#load in datasets and target
if is_debug_mode:
    print("Debug mode is ON!")
    molecule_name = feather.read_dataframe('./data/input/train.feather')['molecule_name'].head(10000).values
else:
    molecule_name = feather.read_dataframe('./data/input/train.feather')['molecule_name'].values
feats = config['feats']
target_name = config['target_name']
train, test = load_datasets(feats, is_debug_mode)
target = load_target(target_name, is_debug_mode)


train_type = train['type'].values
test_type = test['type'].values
logger.debug(feats)

#train.drop(['PropertyFunctor', 'type'], axis=1, inplace=True) #always nan
#test.drop(['PropertyFunctor', 'type'], axis=1, inplace=True) #always nan


if keep_nans:
    # simply keep nans as they are and let the lightgbm handle it
    categorical_cols = list(train.columns[train.dtypes == object])
    logger.debug(categorical_cols) 

else:
    #drop molucules in train with nan descriptors
    #replace nan in test with the mean of train
    nan_cols = list(train.columns.values[train.isnull().any(axis=0)])
    target = target[~train.isnull().any(axis=1)]
    train_type = train_type[~train.isnull().any(axis=1)]
    molecule_name = molecule_name[~train.isnull().any(axis=1)]
    train = train[~train.isnull().any(axis=1)]
    categorical_cols = list(train.columns[train.dtypes == object])
    logger.debug(nan_cols)
    logger.debug(categorical_cols) 

    for col in nan_cols:
        if col in categorical_cols:
            mode = train[col].dropna().mode()
            test[col].fillna(mode[0], inplace=True)
        else:
            median = train[col].dropna().median()
            test[col].fillna(median, inplace=True)


# encoding categorical variables
for col in categorical_cols:
    #le = LabelEncoder()
    print(f'Starting {col}')
    uniques = list(train[col].unique())
    if None in uniques:
        uniques.remove(None)
    mapping = dict(zip(uniques, range(1, len(uniques)+1)))
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

logger.debug(train.shape)
logger.debug(test.shape)

#reeduce memory usage
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()

#train lgbm
params = config['params']
SEED = 42
VAL_SIZE = 0.2
NUM_ROUNDS = 10000

#build models for each type
predictions = np.zeros(len(test))

val_score_list = []

for cur_type in np.unique(train_type):
    cur_type_idx_train = (train_type == cur_type)
    cur_type_idx_test = (test_type == cur_type)

    cur_type_train = train.iloc[cur_type_idx_train]
    cur_type_target = target.iloc[cur_type_idx_train]
    cur_type_mols = molecule_name[cur_type_idx_train]

    train_idx, val_idx = next(GroupShuffleSplit(random_state=SEED, n_splits=1, test_size=VAL_SIZE).split(cur_type_train, cur_type_target, cur_type_mols))

    x_train, y_train = cur_type_train.iloc[train_idx], cur_type_target.iloc[train_idx]
    x_val, y_val = cur_type_train.iloc[val_idx], cur_type_target.iloc[val_idx]

    callbacks = [log_evaluation(logger, period=100)]

    train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_cols)
    val_data = lgb.Dataset(x_val, label=y_val, categorical_feature=categorical_cols)
    clf = lgb.train(params, train_data, NUM_ROUNDS, valid_sets=[train_data, val_data],
                    verbose_eval=False, early_stopping_rounds=100, callbacks=callbacks)
    val_pred = clf.predict(x_val, num_iteration=clf.best_iteration) 
    predictions[cur_type_idx_test] = clf.predict(test.iloc[cur_type_idx_test], num_iteration=clf.best_iteration)

    #store feature importance for this fold
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feats'] = train.columns
    feature_importance_df['importance'] = clf.feature_importance(importance_type='gain')
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(30)
    logger.debug(f'feature importance for {cur_type}')
    logger.debug(feature_importance_df)

    #log val score and feature importance
    val_score = calculate_metric(val_pred, y_val, np.full(len(y_val), cur_type))
    logger.debug(f'val score for {cur_type}: {val_score}')
    val_score_list.append(val_score)


#make submission file
if not is_debug_mode:
    val_score = sum(val_score_list) / len(val_score_list)
    sub = feather.read_dataframe('data/input/sample_submission.feather')
    sub['scalar_coupling_constant'] = predictions
    sub.to_csv(f'data/output/sub_{now}_{val_score:.3f}.csv', index=False)
    logger.debug(f'data/output/sub_{now}_{val_score:.3f}.csv')









