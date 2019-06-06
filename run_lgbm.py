import argparse
import json
import logging
import datetime

from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np
import pandas as pd
import feather

from utils import load_datasets, load_target, get_categorical_feats, calculate_metric, log_evaluation

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
categorical_feats = get_categorical_feats(config['categorical_feats'])
train, test = load_datasets(feats)
target = load_target(target_name)
coupling_types = train['type']
logger.debug(feats)
logger.debug(categorical_feats)
logger.debug(train.shape)


#train lgbm
params = config['params']
SEED = 42
NUM_ROUNDS = 10000

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
callbacks = [log_evaluation(logger, period=100)]

for train_idx, val_idx in kf.split(train, target):
    train_data = lgb.Dataset(train.iloc[train_idx], label=target.iloc[train_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx], categorical_feature=categorical_feats)
    clf = lgb.train(params, train_data, NUM_ROUNDS, valid_sets=[train_data, val_data],
                    verbose_eval=False, early_stopping_rounds=100, callbacks=callbacks)
    oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    #store feature importance for this fold
    fold_importance_df = pd.DataFrame()
    fold_importance_df['feats'] = train.columns
    fold_importance_df['importance'] = clf.feature_importance(importance_type='gain')
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test, num_iteration=clf.best_iteration) / kf.n_splits


#log cv score and feature importance
cv_score = calculate_metric(oof, target.values, coupling_types.values)
logger.debug(cv_score)

feature_importance_df = feature_importance_df.groupby('feats')['importance'].mean().reset_index()
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(10)
logger.debug(feature_importance_df)


#make submission file
sub = feather.read_dataframe('data/input/sample_submission.feather')
sub['scalar_coupling_constant'] = predictions
sub.to_csv(f'data/output/sub_{now}_{cv_score:.3f}.csv', index=False)
logger.debug(f'data/output/sub_{now}_{cv_score:.3f}.csv')









