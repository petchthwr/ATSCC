import torch
import random
import torch.nn.functional as F
from .utils import *
import gc

criterion = torch.nn.CrossEntropyLoss()

# TS2Vec loss
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B

    # remove self-similarities
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # B x 2T x (2T-1)
    logits = -F.log_softmax(logits, dim=-1)  # B x 2T x (2T-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2  #
    return loss

# InfoTS loss
def subsequence_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    s1 = torch.unsqueeze(torch.max(z1, 1)[0], 1)
    s2 = torch.unsqueeze(torch.max(z2, 1)[0], 1)
    loss = instance_contrastive_loss(s1, s2)
    exit(1)
    return loss

def subsequence_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z2 = z2[:,start:start+crop_leng,:]

    crop_z1 = crop_z1.view(B ,k,crop_size,D)
    crop_z2 = crop_z2.view(B ,k,crop_size,D)

    # debug
    # crop_z1 = crop_z1.reshape(B * k, crop_size, D)
    # crop_z2 = crop_z2.reshape(B * k, crop_size, D)
    # return instance_contrastive_loss(crop_z1, crop_z2)+temporal_contrastive_loss(crop_z1,crop_z2)


    if pooling=='max':
        # crop_z1_pooling = torch.max(crop_z1,2)[0]
        # crop_z2_pooling = torch.max(crop_z2,2)[0]
        # crop_z1_pooling = torch.unsqueeze(crop_z1_pooling.view(B*k, D), 1)
        # crop_z2_pooling = torch.unsqueeze(crop_z2_pooling.view(B*k, D), 1)


        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z2 = crop_z2.reshape(B*k,crop_size,D)

        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)
        crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)


    return InfoNCE(crop_z1_pooling,crop_z2_pooling,temperature)

def local_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z1 = crop_z1.view(B ,k,crop_size,D)


    # crop_z2 = z2[:,start:start+crop_leng,:]
    # crop_z2 = crop_z2.view(B ,k,crop_size,D)


    if pooling=='max':
        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2).reshape(B,k,D)

        # crop_z2 = crop_z2.reshape(B*k,crop_size,D)
        # crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        # crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)
    elif pooling=='last':
        crop_z1_pooling = torch.unsqueeze(z1[:,-1,:],1)
        # crop_z2_pooling = torch.unsqueeze(z2[:,-1,:],1)
    elif pooling=='first':
        crop_z1_pooling = torch.unsqueeze(z1[:,0,:],1)
        # crop_z2_pooling = torch.unsqueeze(z2[:,0,:],1)
    else:
        raise NotImplementedError

    crop_z1_pooling_T = crop_z1_pooling.transpose(1,2)

    # B X K * K
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k-1, dtype=torch.float32)
    labels = torch.cat([labels,torch.zeros(1,k-1)],0)
    labels = torch.cat([torch.zeros(k,1),labels],-1)

    pos_labels = labels.cuda()
    pos_labels[k-1,k-2]=1.0


    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0,2]=1.0
    neg_labels[-1,-3]=1.0
    neg_labels = neg_labels.cuda()


    similarity_matrix = similarity_matrices[0]

    # select and combine multiple positives
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss

def global_infoNCE(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)
    elif pooling == 'last':
        z1 = torch.unsqueeze(z1[:,-1,:],1)
        z2 = torch.unsqueeze(z2[:,-1,:],1)
    elif pooling == 'first':
        z1 = torch.unsqueeze(z1[:,0,:],1)
        z2 = torch.unsqueeze(z2[:,0,:],1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature)

def InfoNCE(z1, z2, temperature=1.0):

    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(z1.device)

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss

def hierarchical_snnl(z1, z2, loc_label1, loc_label2, alpha=0.5, temperature=10.0, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * wrapup_local_snnl(z1, z2, loc_label1, loc_label2, temperature=temperature)
        d += 1

        z1, pooled_ind1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2, stride=2, return_indices=True)
        z2, pooled_ind2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2, stride=2, return_indices=True)

        z1 = z1.transpose(1, 2)
        z2 = z2.transpose(1, 2)

        pooled_ind1 = torch.round(torch.mean(pooled_ind1.float().transpose(1,2), dim=-1)).long()
        pooled_ind2 = torch.round(torch.mean(pooled_ind2.float().transpose(1,2), dim=-1)).long()

        batch_indices = torch.arange(pooled_ind1.shape[0]).unsqueeze(-1)  # Create a batch index
        loc_label1 = loc_label1[batch_indices, pooled_ind1]
        loc_label2 = loc_label2[batch_indices, pooled_ind2]

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def soft_nearest_neighbor_loss(x, y, T):

    sim = torch.mm(x, x.t()) / T # Calculate the similarity matrix and scale by temperature
    sim.fill_diagonal_(0) # Remove self-similarity
    masks = torch.eq(y[:, None], y[None, :]).bool() # Create a mask for instances with the same label

    numerator = torch.logsumexp(sim * masks, dim=1)
    denominator = torch.logsumexp(sim * ~masks, dim=1) #* ~masks
    loss = torch.mean(denominator - numerator)

    return loss

def soft_nearest_neighbor_loss_loop(x, y, T):
    loss = 0.0
    n_samples = x.shape[0]

    for i in range(n_samples):
        sim_i = torch.mm(x[i].unsqueeze(0), x.t()).squeeze(0) / T  # (1, E) * (E, N) = (N, )
        sim_i[i] = 0
        mask_i = torch.eq(y[i], y).float() # (N, )
        mask_i[i] = 0
        numerator = torch.logsumexp(sim_i * mask_i, dim=0) # (N, ) * (N, ) = (N, )
        denominator = torch.logsumexp(sim_i, dim=0) # (N, )
        loss_i = denominator - numerator
        loss += loss_i

    return loss / n_samples

def soft_nearest_neighbor_loss_new(x, y, T):

    #x = F.normalize(x, dim=1) # Normalize for cosine similarity
    dist = torch.mm(x, x.t()) # Calculate the similarity matrix and scale by temperature
    dist = torch.sub(torch.tensor(1.0), dist) # Cosine similarity is 1 - cosine distance
    dist.fill_diagonal_(0)  # Remove self-similarity
    dist = dist / T  # Calculate the similarity matrix and scale by temperature
    masks = torch.eq(y[:, None], y[None, :]).float()  # Create a mask for instances with the same label
    ratios = torch.sum(torch.exp(-dist) * masks, dim=1) / torch.clamp(torch.sum(torch.exp(-dist), dim=1), min=1e-6)  # Use clamping to avoid division by zero
    loss = -torch.mean(torch.log(torch.clamp(ratios, min=1e-6)))  # Compute the loss, ensuring numerical stability with clamping

    return loss

def soft_nearest_neighbor_gaussian_loss(x, y, T, min2sigma=20.0):

    sim = torch.mm(x, x.t()) / T # Calculate the similarity matrix and scale by temperature\
    sim.fill_diagonal_(0) # Remove self-similarity
    masks = gaussian_weight_matrix(y, min2sigma=min2sigma)

    numerator = torch.logsumexp(sim * masks, dim=1)
    denominator = torch.logsumexp(sim, dim=1)

    loss = torch.mean(denominator - numerator)

    return loss

def batch_soft_nearest_neighbor_loss(out1, proc_label, temperature=100.0):
    local_loss = []
    for x_local, y_local in zip(out1, proc_label):
        # Filter out valid indices (non-NaN)
        valid_indices = ~torch.isnan(y_local)
        x_local = x_local[valid_indices]
        y_local = y_local[valid_indices]

        # Compute SNNL loss for valid data points
        if len(x_local) > 0 and len(y_local) > 0:
            snnl = soft_nearest_neighbor_loss(x_local, y_local, T=temperature)
            local_loss.append(snnl)  # Append the computed loss to the list

    # Normalize the local loss
    if len(local_loss) > 0:
        local_loss = torch.stack(local_loss).mean()
    else:
        local_loss = torch.tensor(0.0)  # Return zero if there are no valid subsets

    return local_loss

def wrapup_local_snnl(z1, z2, proc_label1, proc_label2, temperature=100.0):
    # Concatenate z1 and z2, proc_label1 and proc_label2 along time dimension
    z = torch.cat((z1, z2), dim=1)
    proc_label = torch.cat((proc_label1, proc_label2), dim=1)
    return batch_soft_nearest_neighbor_loss(z, proc_label, temperature=temperature)

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

def layer_wise_snnl(z, proc_label, temperature=100.0, portion=0.2):

    valid_indices = ~proc_label.reshape(-1).isnan()
    proc_label = flatten_local_label(proc_label)
    z = [z_i.reshape(-1, z_i.shape[-1])[valid_indices] for z_i in z]

    sample_size = int(len(z[0]) * portion)
    indices = [torch.randperm(len(z_i))[:sample_size] for z_i in z]
    z = [z_i[idx] for z_i, idx in zip(z, indices)]

    return sum([soft_nearest_neighbor_loss(z_i, proc_label[idx], temperature) for z_i, idx in zip(z, indices)]) / len(z)

def soft_gaussian_neighborhood_loss(z, proc_label, temperature=100.0):

    valid_indices = ~proc_label.reshape(-1).isnan()
    z = z.reshape(-1, z.shape[-1])[valid_indices]
    proc_label = flatten_local_label(proc_label)

    return soft_nearest_neighbor_gaussian_loss(z, proc_label, temperature, min2sigma=20.0)


