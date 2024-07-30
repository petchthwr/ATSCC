import torch
import random
import torch.nn.functional as F
from .utils import *
import gc

# ATSCC loss
def soft_nearest_neighbor_loss(x, y, T):

    sim = torch.mm(x, x.t()) / T # Calculate the similarity matrix and scale by temperature
    sim.fill_diagonal_(0) # Remove self-similarity
    masks = torch.eq(y[:, None], y[None, :]).bool() # Create a mask for instances with the same label

    numerator = torch.logsumexp(sim * masks, dim=1) # Positive instances
    denominator = torch.logsumexp(sim * ~masks, dim=1) # Negative instances
    loss = torch.mean(denominator - numerator) # If loss is negative, it means that the positive instances are closer than the negative instances

    return loss

def flat_snnl(out1, proc_label, temperature=100.0, portion=0.2):
    valid_indices = ~proc_label.reshape(-1).isnan()

    out1 = out1.reshape(-1, out1.shape[-1])[valid_indices]
    proc_label = flatten_local_label(proc_label)

    sample_size = int(len(out1) * portion)
    indices = torch.randperm(len(out1))[:sample_size]
    out1 = out1[indices]
    proc_label = proc_label[indices]
    proc_label = proc_label.long()

    return soft_nearest_neighbor_loss(out1, proc_label, temperature)
