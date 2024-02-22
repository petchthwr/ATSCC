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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_scaling(x_train, scaler='standard'):
    if scaler == 'standard':
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    return x_train, scaler

def get_directional_vec(data):

    data = data.transpose(0, 2, 1)

    # Calculate velocity as difference between successive time steps
    vel = np.diff(data, axis=-1)
    vel = np.concatenate([vel, vel[:, :, -1:]], axis=-1)

    # Normalize the velocity vectors
    vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_norm = np.where(vel_norm > 1e-9, vel_norm, 1e-9)  # Avoid division by zero
    directional_vec = vel / vel_norm  # Normalized velocity

    return directional_vec.transpose(0, 2, 1)

def get_polar(data):
    data = data.transpose(0, 2, 1)

    # Calculate r and theta from x, y
    r = np.linalg.norm(data[:, :2, :], axis=1, keepdims=True)
    theta = np.arctan2(data[:, :1, :], data[:, 1:2, :]) # arctan(y/x)

    sin = np.sin(theta)
    cos = np.cos(theta)

    polar = np.concatenate([r, sin, cos], axis=1)
    return polar.transpose(0, 2, 1)

def load_ATFM_data(path, split_point='auto', downsample=2, size_lim=None):

    # Load data, takes only the last 3 columns (x, y, z)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x_train = pickle.load(f)[:, -3:, :]
        x_train = np.transpose(x_train, (0, 2, 1))
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)[:, -3:, :]
        x_test = np.transpose(x_test, (0, 2, 1))

    # If data path has ESSA in it, apply Savitzky-Golay filter
    #if 'ESSA' in path:
    #    x_train = savgol_filter_data(x_train, window_length=21, polyorder=1)
    #    x_test = savgol_filter_data(x_test, window_length=21, polyorder=1)

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
    def __init__(self, x, y=None, epsilon=0.05, eval=True, device='cuda', precomputed_rdp=None, polar=False, direction=False, fitted_scaler=None):
        super(ATPCCDataset, self).__init__()
        self.time_series = x
        self.device = device
        self.eval = eval
        train_flag = 'train' if not eval else 'test'
        self.label = y
        self.epsilon = epsilon
        self.polar = polar
        self.direction = direction
        self.resize = tsaug.Resize(size=100)
        self.length_list = [x_i[~np.isnan(x_i).any(axis=1)].shape[0] for x_i in x]

        # RDP precomputation
        if precomputed_rdp is None:
            print('RDP segmentation precomputing...', end=' ')
            self.rdp = [self.get_procedural_label(x, epsilon=epsilon) for x in tqdm(self.time_series, desc=f'RDP precomputing; {train_flag}', leave=True)]
        else:
            self.rdp = precomputed_rdp

        # Feature precomputation
        self.features = x
        if direction:
            directional_features = np.array([self.get_data_directional_vec(x_i) for x_i in self.features])
            self.features = np.concatenate([self.features, directional_features], axis=-1)
        if polar:
            polar_features = np.array([self.get_data_polar(x_i) for x_i in self.features])
            self.features = np.concatenate([self.features, polar_features], axis=-1)
        self.time_series = self.features
        
        """if fitted_scaler is not None:
            self.scaler = fitted_scaler
            self.features = fitted_scaler.transform(self.features.reshape(-1, self.features.shape[-1])).reshape(self.features.shape)
        else:
            self.features, self.scaler = data_scaling(self.features, scaler='minmax')"""

        self.scaler = fitted_scaler

        self.procedural_labels = [x[0] for x in self.rdp]
        self.rdp_mask = [x[1] for x in self.rdp]

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, i):
        x = self.features[i]
        x = x[~np.isnan(x).any(axis=1)]
        features = x
        proc_label = self.procedural_labels[i]

        if self.eval: # If not using augmentation, return the original time series
            return features, self.procedural_labels[i], self.label[i] if self.label is not None else None

        # Our augmentation simplify
        aug1, proc_label1 = x, proc_label
        return aug1, proc_label1

        #size = np.random.choice(self.length_list)
        #self.resize.size = int(size)
        #aug2 = self.resize.augment(aug1)
        #proc_label2 = np.round(self.resize.augment(proc_label1))
        #return aug1, aug2, proc_label1, proc_label2

    def get_procedural_label(self, sample, epsilon=0.05, num_split=25):
        # Input shape of sample: (T, 3)
        # Label shape: (T, )

        sample = sample[~np.isnan(sample).any(axis=1)]
        mask = rdp(sample, epsilon=epsilon, return_mask=True)
        partition_labels = np.cumsum(mask.astype(float))
        partition_labels[-1] = partition_labels[-2]

        return partition_labels, mask

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

        polar = np.concatenate([r, sin_theta, cos_theta], axis=1)

        return polar

    def get_data_directional_vec(self, x):
        original_shape = x.shape
        x = x[~np.isnan(x).any(axis=1)]
        u = self.get_directional_vec(x)
        u = np.pad(u, ((0, original_shape[0] - u.shape[0]), (0, 0)), mode='constant', constant_values=float('nan'))
        return u

    def get_data_polar(self, x):
        original_shape = x.shape
        x = x[~np.isnan(x).any(axis=1)]
        p = self.get_polar(x)
        p = np.pad(p, ((0, original_shape[0] - p.shape[0]), (0, 0)), mode='constant', constant_values=float('nan'))
        return p

def pad_stack_train(batch):

    aug1, proc_label1 = zip(*batch)
    #aug1, aug2, proc_label1, proc_label2 = zip(*batch)

    aug1 = [torch.tensor(t, dtype=torch.float32) for t in aug1]
    proc_label1 = [torch.tensor(t, dtype=torch.float32) for t in proc_label1]
    max_aug1_length = max(t.size(0) for t in aug1)
    max_label1_length = max(t.size(0) for t in proc_label1)
    assert max_aug1_length == max_label1_length, 'Max lengths are not equal!'

    #aug2 = [torch.tensor(t, dtype=torch.float32) for t in aug2]
    #proc_label2 = [torch.tensor(t, dtype=torch.float32) for t in proc_label2]
    #max_aug2_length = max(t.size(0) for t in aug2)
    #max_label2_length = max(t.size(0) for t in proc_label2)
    #assert max_aug2_length == max_label2_length, 'Max lengths are not equal!'

    #max_length = max(max_aug1_length, max_aug2_length)

    max_length = max_aug1_length
    aug1_padded = [F.pad(input=t, pad=(0, 0, 0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in aug1]
    proc_label1_padded = [F.pad(input=t, pad=(0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in proc_label1]
    #aug2_padded = [F.pad(input=t, pad=(0, 0, 0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in aug2]
    #proc_label2_padded = [F.pad(input=t, pad=(0, max_length - t.size(0)), mode='constant', value=float('nan')) for t in proc_label2]
    #return torch.stack(aug1_padded), torch.stack(aug2_padded), torch.stack(proc_label1_padded), torch.stack(proc_label2_padded)

    return torch.stack(aug1_padded), torch.stack(proc_label1_padded)


def pad_stack_test(batch):

    x, proc_label, label = zip(*batch)
    x = [torch.tensor(t, dtype=torch.float32) for t in x]
    proc_label = [torch.tensor(t, dtype=torch.float32) for t in proc_label]
    max_x_length = max(t.size(0) for t in x)
    max_label_length = max(t.size(0) for t in proc_label)
    assert max_x_length == max_label_length, 'Max lengths are not equal!'

    x_padded = [F.pad(input=t, pad=(0, 0, 0, max_x_length - t.size(0)), mode='constant', value=float('nan')) for t in x] # Pad bottom
    proc_label_padded = [F.pad(input=t, pad=(0, max_label_length - t.size(0)), mode='constant', value=float('nan')) for t in proc_label] # Pad right

    if label[0] is None:
        return torch.stack(x_padded), torch.stack(proc_label_padded), None

    return torch.stack(x_padded), torch.stack(proc_label_padded), torch.tensor(label, dtype=torch.long)


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