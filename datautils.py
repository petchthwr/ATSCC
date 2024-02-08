import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data

from rdp import rdp
import tsaug
import numpy as np
import pickle
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from losses.utils import find_split_indices
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def load_ATFM_data(path, split_point='auto', downsample=2, size_lim=None):

    # Load data, takes only the last 3 columns (x, y, z)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x_train = pickle.load(f)[:, -3:, :]
        x_train = np.transpose(x_train, (0, 2, 1))
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)[:, -3:, :]
        x_test = np.transpose(x_test, (0, 2, 1))

    # If data path has ESSA in it, apply Savitzky-Golay filter
    if 'ESSA' in path:
        x_train = savgol_filter_data(x_train, window_length=21, polyorder=1)
        x_test = savgol_filter_data(x_test, window_length=21, polyorder=1)

    # Load labels if they exist
    if not os.path.exists(os.path.join(path, 'y_train.pkl')) or not os.path.exists(os.path.join(path, 'y_test.pkl')):
        y_train, y_test = None, None
    else:
        label_encoder = LabelEncoder()
        with open(os.path.join(path, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(path, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        label_encoder.fit(np.concatenate([y_train, y_test], axis=0))
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        assert x_train.shape[0] == y_train.shape[0], 'Number of samples and labels do not match!'
        assert x_test.shape[0] == y_test.shape[0], 'Number of samples and labels do not match!'

    x_train = x_train[:, ::downsample, :] if downsample != 1 else x_train
    x_test = x_test[:, ::downsample, :] if downsample != 1 else x_test

    if split_point != 'auto':
        x = np.concatenate([x_train, x_test], axis=0)
        x_train, x_test = x[:int(len(x) * split_point)], x[int(len(x) * split_point):]
        if y_train is not None:
            y = np.concatenate([y_train, y_test], axis=0)
            y_train, y_test = y[:int(len(y) * split_point)], y[int(len(y) * split_point):]

    x_train = x_train[:size_lim] if size_lim is not None else x_train
    x_test = x_test[:size_lim] if size_lim is not None else x_test
    y_train = y_train[:size_lim] if size_lim is not None and y_train is not None else y_train
    y_test = y_test[:size_lim] if size_lim is not None and y_test is not None else y_test

    return x_train, x_test, y_train, y_test


class ATPCCDataset(data.Dataset):
    def __init__(self, x, y=None, epsilon=0.05, eval=True, device='cuda', precomputed_rdp=None, polar=False, direction=False):
        super(ATPCCDataset, self).__init__()
        self.time_series = x
        self.device = device
        self.epsilon = epsilon
        self.eval = eval
        self.Resize = tsaug.Resize(size=100)
        self.label = y

        if precomputed_rdp is None:
            print('RDP segmentation precomputing...', end=' ')
            self.rdp = [self.get_procedural_label(x, epsilon=epsilon) for x in tqdm(self.time_series, desc='RDP precomputing', leave=True)]
        else:
            self.rdp = precomputed_rdp

        self.polar = polar
        self.direction = direction
        self.procedural_labels = [x[0] for x in self.rdp]
        self.rdp_mask = [x[1] for x in self.rdp]
        self.point_reduced_time_series = [x[2] for x in self.rdp]
        self.simplified_time_series = [x[3] for x in self.rdp]

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, i):
        x = self.time_series[i]
        x = x[~np.isnan(x).any(axis=1)]
        features = x
        proc_label = self.procedural_labels[i]

        if self.eval: # If not using augmentation, return the original time series

            if self.direction:
                u_features = self.get_directional_vec(features)
                features = np.concatenate([features, u_features], axis=1)

            if self.polar:
                p_features = self.get_polar(features)
                features = np.concatenate([features, p_features], axis=1)

            return features, self.procedural_labels[i], self.label[i] if self.label is not None else None

        # Our augmentation simplify
        aug1, proc_label1 = x, proc_label
        aug2, proc_label2 = self.simplify(i)

        if self.direction:
            u_aug1, u_aug2 = self.get_directional_vec(aug1), self.get_directional_vec(aug2)
            aug1 = np.concatenate([aug1, u_aug1], axis=1)
            aug2 = np.concatenate([aug2, u_aug2], axis=1)

        if self.polar:
            p_aug1, p_aug2 = self.get_polar(aug1), self.get_polar(aug2)
            aug1 = np.concatenate([aug1, p_aug1], axis=1)
            aug2 = np.concatenate([aug2, p_aug2], axis=1)

        """# TS2Vec augmentation
        if self.direction:
            u_features = self.get_directional_vec(features)
            features = np.concatenate([features, u_features], axis=1)
        if self.polar:
            p_features = self.get_polar(features)
            features = np.concatenate([features, p_features], axis=1)
        aug1, proc_label1 = self.crop(features, proc_label, keep_dim=True)
        aug2, proc_label2 = self.crop(features, proc_label, keep_dim=True)"""

        return aug1, aug2, proc_label1, proc_label2

    def get_procedural_label(self, sample, epsilon=0.05, num_split=25):
        # Input shape of sample: (T, 3)
        # Label shape: (T, )

        sample = sample[~np.isnan(sample).any(axis=1)]
        mask = rdp(sample, epsilon=epsilon, return_mask=True)
        partition_labels = np.cumsum(mask.astype(float))
        partition_labels[-1] = partition_labels[-2]

        # Create NaN like sample
        simplified_sample = sample.copy()
        simplified_sample[~mask] = float('nan') # Replace non-partition points with NaN

        # Interpolate to fill NaNs
        for i in range(simplified_sample.shape[1]):
            f = interp1d(np.arange(simplified_sample.shape[0])[~np.isnan(simplified_sample[:, i])], simplified_sample[~np.isnan(simplified_sample[:, i]), i])
            simplified_sample[:, i] = f(np.arange(simplified_sample.shape[0]))

        point_reduced_sample = sample.copy()
        point_reduced_sample[~mask] = float('nan')

        return partition_labels, mask, point_reduced_sample, simplified_sample

    def get_velocity(self, x):
        vel = np.diff(x, axis=0)
        vel = np.concatenate([vel, vel[-1:]], axis=0)
        return vel

    def get_directional_vec(self, x):
        vel = self.get_velocity(x)
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel_norm = np.where(vel_norm > 1e-9, vel_norm, 1e-9) # Replace near-zero norms with 1e-9 to avoid division by zero
        directional_vec = vel / vel_norm # Perform division
        return directional_vec

    def get_polar(self, x):
        # Calculate r and theta from x, y
        r = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
        theta = np.arctan2(x[:, 1:2], x[:, :1])

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        #theta = (theta + np.pi) / (2 * np.pi)

        return np.concatenate([r, sin_theta, cos_theta], axis=1)

    def crop(self, x, proc_label, keep_dim=True):
        # Assuming x is a numpy array of shape [T, Ch]
        ts = x
        cropped_ts = ts.copy()
        cropped_proc_label = proc_label.copy()
        seq_len = x.shape[0]
        crop_l = np.random.randint(low=2, high=seq_len + 1)
        start = np.random.randint(seq_len - crop_l + 1)
        end = start + crop_l
        start = max(0, start)
        end = min(end, seq_len)

        if keep_dim:
            cropped_ts[:start, :] = float('nan')
            cropped_ts[end:, :] = float('nan')
        else:
            cropped_ts = cropped_ts[start:end, :]
            cropped_proc_label = cropped_proc_label[start:end]

        return cropped_ts, cropped_proc_label

    def simplify(self, ind):
        return self.simplified_time_series[ind], self.procedural_labels[ind]

    def point_reduce(self, ind):
        reduced_proc_label = self.procedural_labels[ind].copy()
        mask = self.rdp_mask[ind]
        reduced_proc_label[~mask] = float('nan')
        return self.point_reduced_time_series[ind], reduced_proc_label

def pad_stack_train(batch):

    aug1, aug2, proc_label1, proc_label2 = zip(*batch)
    aug1 = [torch.tensor(t, dtype=torch.float32) for t in aug1]
    aug2 = [torch.tensor(t, dtype=torch.float32) for t in aug2]
    proc_label1 = [torch.tensor(t, dtype=torch.float32) for t in proc_label1]
    proc_label2 = [torch.tensor(t, dtype=torch.float32) for t in proc_label2]
    max_aug1_length = max(t.size(0) for t in aug1)
    max_aug2_length = max(t.size(0) for t in aug2)
    max_label1_length = max(t.size(0) for t in proc_label1)
    max_label2_length = max(t.size(0) for t in proc_label2)
    assert max_aug1_length == max_label1_length, 'Max lengths are not equal!'
    assert max_aug2_length == max_label2_length, 'Max lengths are not equal!'
    max_length = max(max_aug1_length, max_aug2_length)

    # Pad according to max length on the right
    aug1_padded = [F.pad(input=t, pad=(0, 0, max_length - t.size(0), 0), mode='constant', value=float('nan')) for t in aug1]
    aug2_padded = [F.pad(input=t, pad=(0, 0, max_length - t.size(0), 0), mode='constant', value=float('nan')) for t in aug2]
    proc_label1_padded = [F.pad(input=t, pad=(max_length - t.size(0), 0), mode='constant', value=float('nan')) for t in proc_label1]
    proc_label2_padded = [F.pad(input=t, pad=(max_length - t.size(0), 0), mode='constant', value=float('nan')) for t in proc_label2]

    # Pad according to max length on the left
    #aug1_padded = [F.pad(input=t, pad=(0, 0, 0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in aug1]
    #aug2_padded = [F.pad(input=t, pad=(0, 0, 0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in aug2]
    #proc_label1_padded = [F.pad(input=t, pad=(0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in proc_label1]
    #proc_label2_padded = [F.pad(input=t, pad=(0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in proc_label2]

    return torch.stack(aug1_padded), torch.stack(aug2_padded), torch.stack(proc_label1_padded), torch.stack(proc_label2_padded)

def pad_stack_test(batch):

    x, proc_label, label = zip(*batch)
    x = [torch.tensor(t, dtype=torch.float32) for t in x]
    proc_label = [torch.tensor(t, dtype=torch.float32) for t in proc_label]
    max_x_length = max(t.size(0) for t in x)
    max_label_length = max(t.size(0) for t in proc_label)
    assert max_x_length == max_label_length, 'Max lengths are not equal!'

    # Pad according to max length
    x_padded = [F.pad(input=t, pad=(0, 0, max_x_length - t.size(0), 0), mode='constant', value=float('nan')) for t in x]
    proc_label_padded = [F.pad(input=t, pad=(max_label_length - t.size(0), 0), mode='constant', value=float('nan')) for t in proc_label]

    # Pad according to max length on the left
    #x_padded = [F.pad(input=t, pad=(0, 0, 0, max_x_length - t.size(0)), mode='constant', value=float('nan')) for t in x]
    #proc_label_padded = [F.pad(input=t, pad=(0, max_label_length - t.size(0)), mode='constant', value=float('nan')) for t in proc_label]

    if label[0] is None:
        return torch.stack(x_padded), torch.stack(proc_label_padded), None

    return torch.stack(x_padded), torch.stack(proc_label_padded), torch.tensor(label, dtype=torch.long)

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def savgol_filter_data(data, window_length, polyorder):
    """
    Apply Savitzky-Golay filter to a 3D NumPy array with NaN handling along the timestep axis.

    Parameters:
    - data: NumPy array of shape (number of data, timestep, feature).
    - window_length: Length of the filter window. Must be a positive odd integer.
    - polyorder: Order of the polynomial used to fit the samples. Must be less than window_length.

    Returns:
    - filtered_data: NumPy array of the same shape as data, containing the smoothed data.
    """
    print('data shape:', data.shape)
    # Validate window_length and polyorder
    if window_length % 2 == 0 or window_length >= data.shape[1] or polyorder >= window_length:
        raise ValueError("window_length must be a positive odd number smaller than timestep size, and polyorder must be less than window_length")

    # Initialize an array for the filtered data
    filtered_data = np.zeros_like(data)

    for i in range(data.shape[0]):  # Iterate over the number of data points
        for j in range(data.shape[2]):  # Iterate over the features
            y = data[i, :, j]
            # Check if there are NaNs in the sequence
            if np.isnan(y).any():
                # Interpolate to fill NaNs, ignoring them in the interpolation
                x = np.arange(len(y))
                valid = ~np.isnan(y)
                interp_func = interp1d(x[valid], y[valid], kind='linear', bounds_error=False, fill_value="extrapolate")
                y_interp = interp_func(x)
                filtered_data[i, :, j] = savgol_filter(y_interp, window_length, polyorder)
            else:
                # Apply the filter directly if there are no NaNs
                filtered_data[i, :, j] = savgol_filter(y, window_length, polyorder)

    return filtered_data