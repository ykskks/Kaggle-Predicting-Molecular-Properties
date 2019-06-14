import argparse
import json
import logging
import datetime
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np
import pandas as pd
import feather

from utils import load_datasets, load_target, get_categorical_feats, calculate_metric, log_evaluation, reduce_mem_usage

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration for this run")
args = parser.parse_args()
config_path = args.config


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
coupling_types = train['type']
logger.debug(feats)

train.drop('PropertyFunctor', axis=1, inplace=True) #always nan
test.drop('PropertyFunctor', axis=1, inplace=True) #always nan

#drop molucules in train with nan descriptors
#replace nan in test with the mean of train
nan_cols = list(train.columns.values[train.isnull().any(axis=0)])
coupling_types = coupling_types[~train.isnull().any(axis=1)]
target = target[~train.isnull().any(axis=1)]
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

train, val, target_train, target_val = train_test_split(train, target, test_size=VAL_SIZE, 
                                                        random_state=SEED, shuffle=True, stratify=train['type'])
callbacks = [log_evaluation(logger, period=100)]


train_data = lgb.Dataset(train, label=target_train, categorical_feature=categorical_cols)
val_data = lgb.Dataset(val, label=target_val, categorical_feature=categorical_cols)
clf = lgb.train(params, train_data, NUM_ROUNDS, valid_sets=[train_data, val_data],
                verbose_eval=False, early_stopping_rounds=100, callbacks=callbacks)
val_pred = clf.predict(val, num_iteration=clf.best_iteration)

#store feature importance for this fold
feature_importance_df = pd.DataFrame()
feature_importance_df['feats'] = train.columns
feature_importance_df['importance'] = clf.feature_importance(importance_type='gain')


predictions = clf.predict(test, num_iteration=clf.best_iteration) 


#log val score and feature importance
val_score = calculate_metric(val_pred, target_val, val['type'].values)
logger.debug(val_score)

#feature_importance_df = feature_importance_df.groupby('feats')['importance'].mean().reset_index()
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(30)
logger.debug(feature_importance_df)


#make submission file
sub = feather.read_dataframe('data/input/sample_submission.feather')
sub['scalar_coupling_constant'] = predictions
sub.to_csv(f'data/output/sub_{now}_{val_score:.3f}.csv', index=False)
logger.debug(f'data/output/sub_{now}_{val_score:.3f}.csv')









