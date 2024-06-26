from datautils import *
from plotutils import *
from losses.losses import *
from umap import UMAP
import tasks
from pathlib import Path
import warnings
import pickle
import time
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def reproducibility(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def scoredict_to_str(score):
    score_str = ''
    for key, val in score.items():
        score_str += f'{key}: {val:.6f} '
    return score_str


def data_to_path(data):
    if data == 'RKSIa':
        return 'data/RKSI/arrival'
    elif data == 'RKSId':
        return 'data/RKSI/departure'
    elif data == 'ESSA':
        return 'data/ESSA/arrival'
    elif data == 'LSZH':
        return 'data/LSZH/arrival'
    elif data == 'RKSIa_v':
        return 'data/RKSI/arrival_v'
    elif data == 'RKSId_v':
        return 'data/RKSI/departure_v'
    elif data == 'ESSA_v':
        return 'data/ESSA/arrival_v'
    elif data == 'LSZH_v':
        return 'data/LSZH/arrival_v'
    else:
        raise ValueError('Invalid data')

def load_data(dataset, split_point, downsample=2, size_lim=None, rdp_epsilon=0.005, batch_size=32, device='cuda:1', direction=False, polar=False, shuffle=True):

    datapath = data_to_path(dataset)
    x_train, x_test, y_train, y_test = load_ATFM_data(datapath, split_point, downsample=downsample, size_lim=size_lim)

    train_dataset = ATPCCDataset(x_train, y_train, rdp_epsilon, False, device, direction=direction, polar=polar)
    test_dataset = ATPCCDataset(x_test, y_test, rdp_epsilon, True, device, direction=direction, polar=polar, fitted_scaler=train_dataset.scaler)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_stack_train, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_stack_test, pin_memory=True, num_workers=4)

    return train_loader, test_loader


def apply_pooling(out, proc_label, pooling='mean'):
    """
    Apply specified pooling strategy to extract representations from a tensor.

    Parameters:
    - out: Tensor of shape [batch_size, seq_len, features], output from a model.
    - proc_label: Tensor of shape [batch_size, seq_len], used to identify valid indices (non-NaN).
    - pooling: String, specifies the pooling strategy ('max', 'last', 'first', 'mean').

    Returns:
    - cls: Tensor of pooled representations.
    """
    cls = []
    for i in range(out.size(0)):
        valid_indices = ~proc_label[i].isnan()
        if pooling == 'max':
            pooled = out[i, valid_indices].max(dim=0).values
        elif pooling == 'last':
            pooled = out[i, valid_indices][-1]
        elif pooling == 'first':
            pooled = out[i, valid_indices][0]
        elif pooling == 'mean':
            pooled = out[i, valid_indices].mean(dim=0)
        else:
            raise ValueError("Unsupported pooling type. Choose from 'max', 'last', 'first', 'mean'.")
        cls.append(pooled)

    cls = torch.stack(cls)
    return cls


def encode(model, loader, device, pooling='last'):

    org_training = model.training
    model.eval()
    loader.dataset.eval = True

    with torch.no_grad():
        output, traj, full_encode, label_list = [], [], [], []
        split_ind = []
        for batch, proc_label, label in loader:

            x = batch.to(device)
            traj.append(x.cpu().numpy())
            out = model(x)

            # Full encoding
            full_encoding = out
            full_encode.append(full_encoding.cpu().numpy())

            # Representation
            cls = apply_pooling(out, proc_label, pooling=pooling)
            output.append(cls)

            split_ind.extend([find_split_indices(label[~label.isnan()]) for label in proc_label])

            # Label
            if label is None:
                label_list = None
            else:
                label_list.append(label)

        max_T = max([t.shape[1] for t in traj])
        traj = np.vstack([np.pad(t, ((0, 0), (0, max_T - t.shape[1]), (0, 0)), 'constant', constant_values=np.nan) for t in traj])
        full_encode = np.vstack([np.pad(t, ((0, 0), (0, max_T - t.shape[1]), (0, 0)), 'constant', constant_values=np.nan) for t in full_encode])
        output = torch.cat(output, dim=0)
        label_list = torch.cat(label_list, dim=0).cpu().numpy() if label_list is not None else None

    model.train(org_training)

    return output.cpu().numpy(), traj, full_encode, split_ind, label_list


def get_loader_label(loader):
    labels = []
    for _, _, label in loader:

        # if label is none exit loop and return none
        if label is None:
            return None

        labels.append(label)
    return torch.cat(labels, dim=0).cpu().numpy()


def evaluate(train_loader, test_loader, Encoder, device, epoch, datapath, clus_model, pooling='last', eval_method='clustering', visualize=False):
    # If path does not exist, create it
    Path(f'figures/{datapath}/UMAP').mkdir(parents=True, exist_ok=True)
    Path(f'figures/{datapath}/trajectories').mkdir(parents=True, exist_ok=True)
    Path(f'figures/{datapath}/clustered').mkdir(parents=True, exist_ok=True)
    Path(f'figures/{datapath}/dendrogram').mkdir(parents=True, exist_ok=True)
    Path(f'figures/{datapath}/scores').mkdir(parents=True, exist_ok=True)

    ori_train_loader_eval = train_loader.dataset.eval
    ori_train_collate_fn = train_loader.collate_fn
    train_loader.dataset.eval = True
    train_loader.collate_fn = pad_stack_test

    train_repr, train_traj, train_full_encode, split_ind_train, train_label = encode(Encoder, train_loader, device, pooling=pooling)
    test_repr, test_traj, test_full_encode, split_ind_test, test_label = encode(Encoder, test_loader, device, pooling=pooling)

    # Inverse transform the data traj
    test_traj = test_loader.dataset.time_series

    # Export test representations as pickle
    Path(f'figures/{datapath}/test_repr').mkdir(parents=True, exist_ok=True)
    with open(f'figures/{datapath}/test_repr/test_repr_epoch_{epoch}.pkl', 'wb') as f:
        pickle.dump(test_repr, f)

    if visualize:
        if test_label is not None:
            index_to_sample = range(len(test_full_encode))[::50]
            index_to_sample = sorted(index_to_sample, key=lambda x: test_label[x])
        else:
            index_to_sample = range(len(test_full_encode))[::50]
        for i in index_to_sample:
            visualize_encoding_ADSB(test_full_encode[i], test_traj[i][~np.isnan(test_traj[i]).all(axis=1)], split_ind_test[i], datapath, i)

    n_cluster = len(set(test_label)) if test_label is not None else 16
    clus_model.n_clusters = n_cluster
    umap = UMAP(n_components=2)

    if test_label is None:
        raise ValueError('Test labels are not provided')

    out, eval_res = tasks.eval_classification(train_repr, train_label, test_repr, test_label, eval_protocol='svm')
    acc = eval_res['acc']
    auprc = eval_res['auprc']

    cluster_assignments = clus_model.fit_predict(test_repr)

    if test_label is not None:
        max_nmi, max_ari, max_mi, best_num_clusters_nmi, best_num_clusters_ari, best_num_clusters_mi = (
            calculate_NMI_ARI(clus_model, test_repr, Path(f'figures/{datapath}/scores') / f'scores_epoch_{epoch}.png', test_label))
        score = {'NMI': max_nmi, 'ARI': max_ari, 'MI': max_mi, 'ACC': acc, 'AUPRC': auprc}
        clus_model.n_clusters = best_num_clusters_nmi
    else:
        score = None
        ValueError('Test labels are not provided')

    if visualize:
        umap_result = umap.fit_transform(test_repr)
        plot_umap_embeddings(umap_result, test_label, Path(f'figures/{datapath}/UMAP') / f'UMAP_true_epoch_{epoch}.png')
        plot_umap_embeddings(umap_result, cluster_assignments, Path(f'figures/{datapath}/UMAP') / f'UMAP_epoch_{epoch}.png')
        plot_2d_trajectories(test_traj, cluster_assignments, Path(f'figures/{datapath}/trajectories') / f'Trajectories_epoch_{epoch}.png')
        plot_clustered_trajectories(test_traj, cluster_assignments, Path(f'figures/{datapath}/clustered') / f'Clustered_Trajectories_epoch_{epoch}.png')

    train_loader.dataset.eval = ori_train_loader_eval
    train_loader.collate_fn = ori_train_collate_fn

    return score

def epoch_run(Encoder, loader, device, optim, scheduler, local_temp, current_iter, max_iter):

    epoch_loss = 0.0
    Encoder.train()

    iter_elapsed = 0
    for aug1, proc_label1 in loader:

        aug1 = aug1.to(device)
        proc_label1 = proc_label1.to(device)
        out1 = Encoder(aug1)
        #out1 = Encoder.final_projection(out1)
        loss = flat_snnl(out1, proc_label1, temperature=local_temp, portion=1.0) #if not isinstance(out1, list) else (layer_wise_snnl(out1, proc_label1, temperature=local_temp, portion=1.0))

        if current_iter < max_iter:
            optim.zero_grad()
            loss.backward()
            optim.step()

            if scheduler is not None:
                scheduler.step()

            current_iter += 1
            iter_elapsed += 1
        else:
            break

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / iter_elapsed
    return epoch_loss, Encoder, current_iter


def fit(Encoder, train_loader, test_loader, optim, scheduler, num_epochs, max_iter, eval_every, local_temp, device, data, clus_model, pooling='last', eval_method='clustering', verbose=True, visualize=False):
    loss_log, score_log = [], []
    datapath = data_to_path(data)

    if not os.path.exists(os.path.join(f'ckpt/{datapath}')):
        os.makedirs(os.path.join(f'ckpt/{datapath}'))

    current_iter = 0
    num_epochs = int(num_epochs)
    for epoch in range(num_epochs):
        st = time.time()
        epoch_loss, Encoder, current_iter = epoch_run(Encoder, train_loader, device, optim, scheduler, local_temp, current_iter, max_iter)  # Train model
        loss_log.append(epoch_loss)  # Log loss

        if epoch % eval_every == 0 or epoch == num_epochs - 1 or current_iter >= max_iter:
            epoch = epoch + 1 if epoch == num_epochs - 1 else epoch  # Epoch num correction
            torch.save(Encoder.state_dict(), os.path.join(f'ckpt/{datapath}/encoder_epoch_{epoch}.pt'))
            score = evaluate(train_loader, test_loader, Encoder, device, epoch, datapath, clus_model, pooling=pooling, eval_method=eval_method, visualize=visualize)
            score_log.append(score)
            print(f'Epoch: {epoch}, Iter: {current_iter} ==> Loss: {epoch_loss:.6f} {scoredict_to_str(score)}')

        if epoch % 10 == 0 and verbose:
            et = time.time()
            print(f'Epoch: {epoch} ==> Loss: {epoch_loss:.6f} Time elapsed: {et - st:.2f} seconds')

        if current_iter >= max_iter:
            break

    return loss_log, score_log


"""
def compute_sampling_loss(out, label, temperature, times_sampling, num_sampling, device):
    # Pad the outputs and labels
    max_length = max([o.size(1) for o in out])  # Find the maximum length
    label = [torch.nn.functional.pad(p, (0, max_length - p.size(1)), mode='constant', value=float('nan')) for p in label]  # Pad each tensor
    label = torch.concat(label, dim=0)
    out = [torch.nn.functional.pad(o, (0, 0, max_length - o.size(1), 0), mode='constant', value=float('nan')) for o in out]  # Pad each tensor
    out = torch.concat(out, dim=0)

    # Rearrage to shape (B x T, C)
    valid_indices = ~label.reshape(-1).isnan()
    out = out.reshape(-1, out.size(-1))[valid_indices]
    label = flatten_local_label(label)

    loss = 0.0
    ind = torch.randperm(out.size(0), device=device)[:times_sampling]
    for i in ind:
        # Find positive index by matching the label at index i
        pos_indices = torch.where(label == label[i])[0]
        pos_samples = out[pos_indices]
        pos_labels = label[pos_indices]

        # For negative indices
        neg_indices = torch.where(label != label[i])[0]
        neg_samples = out[neg_indices]
        neg_labels = label[neg_indices]

        # Random sampling for negatives
        if neg_samples.size(0) > num_sampling:
            neg_indices = torch.randperm(neg_samples.size(0), device=device)[:num_sampling]
            neg_samples = neg_samples[neg_indices]
            neg_labels = neg_labels[neg_indices]

        out_i = torch.cat([pos_samples, neg_samples], dim=0)
        label_i = torch.cat([pos_labels, neg_labels], dim=0)

        snnl = soft_nearest_neighbor_loss(out_i, label_i, temperature)
        loss += snnl

    loss /= len(ind)

    return loss

def rearrange_out(batch_by_layer):
    layer_by_batch = list(zip(*batch_by_layer))
    return [list(layer) for layer in layer_by_batch]

def epoch_run_sampling(Encoder, loader, device, optim, temperature, num_sampling=5000, times_sampling=32):
    Encoder.train()

    out = []  # List to store the outputs
    label = []  # List to store the processed labels

    for aug1, _, proc_label1, _ in loader:

        # Send to device
        aug1 = aug1.to(device)
        proc_label1 = proc_label1.to(device)

        # Generate multiple samples
        out1 = Encoder(aug1)

        out.append(out1)
        label.append(proc_label1)

    del aug1, proc_label1
    torch.cuda.empty_cache()

    if isinstance(out[0], list):
        out = rearrange_out(out)
        loss = 0.0
        for o in out:
            loss += compute_sampling_loss(o, label, temperature, times_sampling, num_sampling, device)
    else:
        loss = compute_sampling_loss(out, label, temperature, times_sampling, num_sampling, device)

    optim.zero_grad()
    loss.backward()
    optim.step()
    epoch_loss = loss.item()

    return epoch_loss, Encoder
"""