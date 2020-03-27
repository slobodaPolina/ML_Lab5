import pandas as pd
import numpy as np


# features and values, but numeric class_numeric
def read_data(path):
    df = pd.read_csv(path)
    # вместо букв P/N ставим +-1 для удобства
    df['class_numeric'] = df['class'].apply(lambda value: {'P': 1, 'N': -1}[value])
    features = df[['x', 'y']].to_numpy()
    values = df['class_numeric'].to_numpy()
    return features, values


def split_indices_data(n, batches_number=5):
    # array from 0 to n
    ids = np.arange(n)
    np.random.shuffle(ids)
    # split it into array of batches_number arrays
    return np.array_split(ids, batches_number)


# из ids_batches выбираю все, что не под номером test_num
def train_dataset(features, values, ids_batches, test_num):
    train_ids = np.array([], dtype=np.int64)
    for i in range(len(ids_batches)):
        if i != test_num:
            train_ids = np.concatenate((train_ids, ids_batches[i]), axis=0)
    return features[train_ids], values[train_ids]

