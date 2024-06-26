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
