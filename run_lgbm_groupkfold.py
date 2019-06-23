import argparse
import json
import logging
import datetime
import gc

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np
import pandas as pd
import feather

from utils import load_datasets, load_target, get_categorical_feats, calculate_metric, log_evaluation, reduce_mem_usage

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration for this run")
parser.add_argument("--debug", action="store_true", help="activate debug mode")
args = parser.parse_args()
config_path = args.config
is_debug_mode = args.debug


#get config fot this run
with open(config_path, 'r') as f:
    config = json.load(f)


#prepare the log file
now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
sc = logging.StreamHandler()
logger.addHandler(sc)
fh = logging.FileHandler(f'logs/log_{now}.log')
logger.addHandler(fh)
logger.debug(f'logs/log_{now}.log')
logger.debug(config_path)


#load in datasets and target
feats = config['feats']
target_name = config['target_name']
train, test = load_datasets(feats)
target = load_target(target_name)
molecule_name = feather.read_dataframe('./data/input/train.feather')['molecule_name'].values

if is_debug_mode:
    print("Debug mode is ON!")
    train = train.iloc[:1000]
    test = test.iloc[:1000]
    target = target.iloc[:1000]
    molecule_name = molecule_name[:1000]

train_type = train['type'].values
test_type = test['type'].values
logger.debug(feats)

train.drop(['PropertyFunctor', 'type'], axis=1, inplace=True) #always nan
test.drop(['PropertyFunctor', 'type'], axis=1, inplace=True) #always nan

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

for col in categorical_cols:
    le = LabelEncoder()
    print(f'Starting {col}')
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

logger.debug(train.shape)
logger.debug(test.shape)

#reeduce memory usage
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()

#train lgbm
params = config['params']
SEED = 42
VAL_SIZE = 0.4
NUM_ROUNDS = 10000

#build models for each type
predictions = np.zeros(len(test))

cv_score_list = []

for cur_type in np.unique(train_type):
    cur_type_idx_train = (train_type == cur_type)
    cur_type_idx_test = (test_type == cur_type)
    oof = np.zeros(len(train.iloc[cur_type_idx_train]))
    feature_importance_df = pd.DataFrame()

    kf = GroupKFold(n_splits=2)
    cur_type_train = train.iloc[cur_type_idx_train]
    cur_type_target = target.iloc[cur_type_idx_train]

    for train_idx, val_idx in kf.split(cur_type_train, cur_type_target, groups=molecule_name[cur_type_idx_train]):
        x_train, y_train = cur_type_train.iloc[train_idx], cur_type_target.iloc[train_idx]
        x_val, y_val = cur_type_train.iloc[val_idx], cur_type_target.iloc[val_idx]

        callbacks = [log_evaluation(logger, period=100)]

        train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_cols)
        val_data = lgb.Dataset(x_val, label=y_val, categorical_feature=categorical_cols)
        clf = lgb.train(params, train_data, NUM_ROUNDS, valid_sets=[train_data, val_data],
                        verbose_eval=False, early_stopping_rounds=100, callbacks=callbacks)
        oof[val_idx] = clf.predict(x_val, num_iteration=clf.best_iteration) 
        predictions[cur_type_idx_test] = clf.predict(test.iloc[cur_type_idx_test], num_iteration=clf.best_iteration) / kf.n_splits

        #store feature importance for this fold
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feats'] = train.columns
        fold_importance_df['importance'] = clf.feature_importance(importance_type='gain')
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #average and get the mean feature importance 
    feature_importance_df = feature_importance_df.groupby('feats')['importance'].mean().reset_index().sort_values(by='importance', ascending=False).head(30)
    logger.debug(f'feature importance for {cur_type}')
    logger.debug(feature_importance_df)

    #log val score and feature importance
    cv_score = calculate_metric(oof, cur_type_target, np.full(len(cur_type_train), cur_type))
    logger.debug(f'CV score for {cur_type}: {cv_score}')
    cv_score_list.append(cv_score)


#make submission file
if not is_debug_mode:
    cv_score = sum(cv_score_list) / len(cv_score_list)
    sub = feather.read_dataframe('data/input/sample_submission.feather')
    sub['scalar_coupling_constant'] = predictions
    sub.to_csv(f'data/output/sub_{now}_{cv_score:.3f}.csv', index=False)
    logger.debug(f'data/output/sub_{now}_{cv_score:.3f}.csv')









