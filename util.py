import numpy as np
import pandas as pd
import os.path
import arglist
import pickle


def encode_onehot():
    # For now, they don't have to be processed. Because those consist of 1 and 0
    pass


def normalize_columns_dataframe(df, list_columns_with_min_max):
    result = df.copy()
    ll = np.asarray(list_columns_with_min_max)
    columns_with_min_and_max = pd.DataFrame(list_columns_with_min_max, columns=['name', 'min_value', 'max_value'])

    l_columns_numerical = ll[:, 0].tolist()

    for feature_name in l_columns_numerical:
        min_value = int(columns_with_min_and_max[columns_with_min_and_max.name == feature_name]['min_value'])
        max_value = int(columns_with_min_and_max[columns_with_min_and_max.name == feature_name]['max_value'])
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def dump_pickle_even_when_no_exist(file, content, dir=None):
    if dir is not None:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    if not dir.endswith(os.path.sep):
        dir += os.path.sep

    # if os.path.isfile(dir+file):
    #     pass
    # else:
    with open(arglist.pickle_dir+arglist.pickle_file, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file, dir=None):
    full_path = None
    r = None
    if dir is None:
        full_path = file
    else:
        if not dir.endswith(os.path.sep):
            dir += os.path.sep
        full_path = dir + file
    with open(full_path, 'rb') as handle:
        r = pickle.load(handle)
    return r