import numpy as np
import pandas as pd

def load_ATFM(dset_name, mode, path):
    """
    Loads the dataset from TSV files, handling NaN values, and returns the data and labels.

    Parameters:
    - dset_name: String, the base name for the TSV files
    - mode: String, typically 'TRAIN' or 'TEST'
    - path: String, the directory path where files are stored

    Returns:
    - data: Numpy array of shape (N, T, 3), with NaN values preserved
    - labels: Numpy array of shape (N,)
    """
    variables = ['X', 'Y', 'Z']
    data = []
    labels = None

    for var in variables:
        tsv_filename = f'{path}/{dset_name}_{mode}_{var}.tsv'
        df = pd.read_csv(tsv_filename, sep='\t', header=None, na_values='NaN')
        if labels is None:
            labels = df.values[:, 0]  # Assumes labels are the first column and only need to be read once
        var_data = df.values[:, 1:]
        data.append(var_data)

    data = np.stack(data, axis=-1)

    return data, labels.astype(int)