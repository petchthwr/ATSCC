import torch

def find_split_indices(labels):
    """
    Find the indices in the tensor where the value changes.

    :param labels: Tensor of labels.
    :return: Tensor of indices where the label changes.
    """
    # Find places where the value changes
    diffs = labels[1:] != labels[:-1]
    # Convert boolean tensor to indices
    split_indices = torch.where(diffs)[0] + 1
    split_indices = torch.cat((torch.tensor([0]).to(split_indices.device), split_indices, torch.tensor([len(labels) - 1]).to(split_indices.device))).to(torch.int64)
    return split_indices

def flatten_local_label(proc_label):
    current_label = 1
    flattened_labels = []
    device = proc_label.device

    # Convert to a list of tensors if it's a 2D tensor
    if proc_label.ndim == 2:
        proc_label = list(proc_label)

    for batch in proc_label:
        label_map = {}
        new_batch = []
        valid_indices = ~batch.isnan()
        batch = batch[valid_indices]
        for label in batch.cpu().numpy():
            if label not in label_map:
                label_map[label] = current_label
                current_label += 1
            new_batch.append(label_map[label])

        flattened_labels.extend(new_batch)

    return torch.tensor(flattened_labels, device=device)

"""def create_row_gaussian_weight(size, center_index, sigma=1, device=None):
    row = torch.arange(size, dtype=torch.long, device=device)
    center_index = torch.tensor(center_index, dtype=torch.long, device=device)
    gaussian_weights = torch.exp(-((row - center_index) ** 2) / (2 * sigma ** 2))
    gaussian_weights[center_index] = 0

    # Normalize the weights
    gaussian_weights /= torch.sum(gaussian_weights)

    return gaussian_weights"""

"""def find_closest_indices(split_indices, target_index):
    # Calculate the absolute differences
    differences = torch.abs(split_indices - target_index)
    # Find the minimum difference
    min_difference_index = torch.argmin(differences)
    return split_indices[min_difference_index]"""

"""def gaussian_weight_matrix(proc_label, min2sigma=20.0):
    # Find the indices where the labels change
    split_indices = find_split_indices(proc_label).to(proc_label.device)

    # Find the closest index in split_indices to each index
    closest_inds = torch.stack([find_closest_indices(split_indices, i) for i in range(len(proc_label))]).to(proc_label.device)

    # Calculate the two sigma values
    two_sigma = torch.abs(torch.arange(len(proc_label), device=proc_label.device) - closest_inds)
    two_sigma = torch.max(two_sigma, torch.ones_like(two_sigma) * min2sigma)

    # Create the Gaussian weight matrix
    gaussian_weight_rows = [create_row_gaussian_weight(len(proc_label), i, sigma=two_sigma[i]/2, device=proc_label.device) for i in range(len(proc_label))]
    gaussian_weight_matrix = torch.stack(gaussian_weight_rows)

    # Weight should be sum to two_sigma
    gaussian_weight_matrix = gaussian_weight_matrix * two_sigma[:, None]

    return gaussian_weight_matrix"""