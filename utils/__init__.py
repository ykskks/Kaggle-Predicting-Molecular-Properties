import time
from contextlib import contextmanager
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import feather
from lightgbm.callback import _format_eval_result 

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'{name}: finished in {end_time - start_time} s')


def load_datasets(feats):
    train_feats = [feather.read_dataframe(f'features/{feat}_train.feather') for feat in feats]
    train = pd.concat(train_feats, axis=1)
    test_feats = [feather.read_dataframe(f'features/{feat}_test.feather') for feat in feats]
    test = pd.concat(test_feats, axis=1)
    return train, test


def load_target(target_name):
    train = feather.read_dataframe('data/input/train.feather')
    target = train[target_name]
    return target


def get_categorical_feats(feats):
    categorical_feats = []
    train, test = load_datasets(feats)

    for col in train.columns:
        categorical_feats.append(col)

    return categorical_feats


def calculate_metric(predicted_values, true_values, coupling_types, floor=1e-9):
    """Inputs should be in numpy.array format"""
    metric = 0
    for coupling_type in np.unique(coupling_types):
        group_metric = np.log(mean_absolute_error(true_values[coupling_types == coupling_type], predicted_values[coupling_types == coupling_type]))
        metric += max(group_metric, floor)
    return metric / len(np.unique(coupling_types))


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback