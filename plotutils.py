import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
import seaborn as sns
from matplotlib import gridspec
from sklearn.decomposition import PCA
import pandas as pd
import os
import warnings
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mutual_info_score

def plot_umap_embeddings(umap_result, cluster_assignments, figure_name):
    plt.figure(figsize=(10, 10))
    unique_clusters = np.unique(cluster_assignments)
    color_map = matplotlib.colormaps['tab20']

    for i in unique_clusters:
        if i == -1:
            color = 'black'
        else:
            color = color_map(i / len(unique_clusters))  # Normalize color index
        plt.scatter(umap_result[cluster_assignments == i, 0],
                    umap_result[cluster_assignments == i, 1],
                    color=color,
                    label='Cluster ' + str(i) if i != -1 else 'Noise')

    plt.legend()
    plt.title('UMAP Embeddings')
    plt.savefig(figure_name)
    plt.close()


def plot_2d_trajectories(trajectories, cluster_assignments, figure_name):
    unique_clusters = np.unique(cluster_assignments)
    legend_handles = []  # List to store legend handles
    color_map = matplotlib.colormaps['tab20']

    plt.figure(figsize=(10, 10))  # Square figure
    for cluster in unique_clusters:
        # Select trajectories belonging to the current cluster
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        cluster_trajectories = trajectories[cluster_indices]

        if cluster == -1:
            color = 'black'
            label = 'Noise'
        else:
            color = color_map(cluster / len(unique_clusters))  # Normalize color index
            label = f'Cluster {cluster + 1}'

        # Create a line object for the legend
        line = plt.Line2D([0], [0], color=color, lw=2, label=label)
        legend_handles.append(line)

        for trajectory in cluster_trajectories:
            xs = trajectory[:, 0]
            ys = trajectory[:, 1]
            plt.plot(xs, ys, color=color, alpha=0.4)

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('2D Top View of Aircraft Trajectories')
    plt.legend(handles=legend_handles, loc='best')  # Add the legend to the plot
    plt.axis('equal')  # Set the aspect of the plot to be equal

    plt.savefig(figure_name)  # Save the figure
    plt.close()


def plot_clustered_trajectories(traj, cluster_assignments, figure_name):
    # Calculate the size of each cluster
    cluster_sizes = {cluster: sum(cluster_assignments == cluster) for cluster in set(cluster_assignments)}

    # Sort clusters based on their size in descending order
    sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)

    num_clusters = len(sorted_clusters)
    cols = int(math.ceil(math.sqrt(num_clusters)))
    rows = int(math.ceil(num_clusters / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    axs = axs.flatten()
    cmap = matplotlib.colormaps['tab20']

    for i, cluster in enumerate(sorted_clusters):
        ax = axs[i]
        c = 'black' if cluster == -1 else cmap(cluster / len(set(cluster_assignments)))
        title = 'Noise' if cluster == -1 else f'Cluster {cluster + 1}'

        cluster_trajectories = traj[cluster_assignments == cluster]
        for trajectory in cluster_trajectories:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=c, alpha=0.4)

        ax.set_title(title, fontsize=18)
        circle = plt.Circle((0, 0), 1.0, color='r', fill=False)
        ax.add_patch(circle)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.grid(True)
        ax.set_aspect('equal')

    for i in range(num_clusters, rows * cols):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close(fig)


def plot_dendrogram(model, truncate_mode=None, p=12, leaf_rotation=90., leaf_font_size=12, save_path=None):
    """
    Plots the dendrogram of a hierarchical clustering model.

    :param model: Hierarchical clustering model (e.g., from sklearn.cluster).
    :param truncate_mode: Truncation mode for the dendrogram.
    :param p: The p parameter for truncate_mode.
    :param leaf_rotation: Rotation of the x-axis labels.
    :param leaf_font_size: Font size of the x-axis labels.
    :param save_path: Path to save the dendrogram plot.
    """
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the dendrogram
    plt.figure(figsize=(15, 10))
    dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p, leaf_rotation=leaf_rotation, leaf_font_size=leaf_font_size, show_contracted=True)

    # Annotating the number of points in each cluster
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index of point or cluster')
    plt.ylabel('Distance')

    if save_path:
        plt.savefig(save_path)


def visualize_encoding_ADSB(encodings, traj, split_ind, path, idx):

    if not os.path.exists(os.path.join(f'figures/{path}/encoding')):
        os.makedirs(os.path.join(f'figures/{path}/encoding'))

    path = path + '/encoding'

    # Reshape encodings and traj
    traj = traj.transpose(1, 0)

    # Drop NaN from encodings
    encodings = encodings[~np.isnan(encodings).any(axis=1)]
    encodings = encodings / np.linalg.norm(encodings, axis=1, keepdims=True)

    f = plt.figure(figsize=(30, 10.5))
    # Create a gridspec object with 2 rows and 2 columns
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 2])

    # Create a list to hold the axes objects
    axs = []

    # Assign subplots to specific positions in the gridspec and add them to the list
    axs.append(plt.subplot(gs[:, 0]))  # First subplot, full left
    axs.append(plt.subplot(gs[0, 1]))  # Second subplot, top right
    axs.append(plt.subplot(gs[1, 1]))  # Third subplot, bottom right

    # Plot the first subplot as a line plot
    axs[0].plot(traj[0, :], traj[1, :])  # Use line plot
    axs[0].scatter(traj[0, split_ind], traj[1, split_ind], color='r', s=100, marker='x')
    axs[0].grid(True)
    axs[0].set_xlabel('X', fontsize=16)
    axs[0].set_ylabel('Y', fontsize=16)
    axs[0].set_title('2D Topview Aircraft Trajectory', fontsize=24, fontweight='bold')
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(-1, 1) # Set x-axis limits
    axs[0].set_ylim(-1, 1) # Set aspect ratio to make it square
    circle = plt.Circle((0, 0), 1.0, color='r', fill=False)
    axs[0].add_patch(circle)

    # Plot the second subplot as a line plot
    features = ['x', 'y', 'z', 'ux', 'uy', 'uz', 'r', 'sin_theta', 'cos_theta']
    for feat in range(min(traj.shape[0], traj.shape[1])):
        axs[1].plot(np.arange(traj.shape[1]), traj[feat], label=features[feat])

    # Plot the vertical lines at the split indices
    for split in split_ind:
        axs[1].axvline(x=split, color='k', linestyle='--', linewidth=1)

    axs[1].set_title('Aircraft Trajectory States', fontsize=24, fontweight='bold')
    axs[1].set_xlabel('Time', fontsize=16)
    axs[1].set_ylabel('Value', fontsize=16)
    axs[1].tick_params(axis='both', labelsize=16)
    axs[1].grid(False)
    axs[1].legend(fontsize=16)
    axs[1].set_aspect('auto')

    # Plot the third subplot as a heatmap
    sns.heatmap(encodings.T, cbar=False, cmap='viridis')
    for split in split_ind[:-1]:
        axs[2].axvline(x=split, color='k', linestyle='--', linewidth=1)

    axs[2].set_title('Encoded Trajectory', fontsize=24, fontweight='bold')
    axs[2].set_ylabel('Encoding dimensions', fontsize=16)
    axs[2].set_xlabel('Encoding windows', fontsize=16)
    axs[2].tick_params(axis='both', labelsize=8)
    axs[2].set_aspect('auto')
    f.tight_layout()
    plt.savefig(os.path.join("./figures/%s" % path, f"embedding_trajectory_hm_{idx}.png"))

    # Close the figure
    plt.close(f)

    # Plot the PCA embedding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(encodings)
        d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'time':np.arange(len(embedding))}
        df = pd.DataFrame(data=d)
        fig, ax = plt.subplots()
        ax.set_title("Trajectory")
        sns.scatterplot(x="f1", y="f2", data=df, hue="time")
        plt.savefig(os.path.join("./figures/%s" % path, f"embedding_trajectory_{idx}.png"))

        # Close the figure
        plt.close(fig)


def encode(sample, encoder, device):
    sample.to(device)
    encoder.to(device)
    with torch.no_grad():
        output = encoder(sample)
    return output


def calculate_NMI_ARI(clus_model, to_be_fit, filename, true_label=None):

    nmi = []
    ari = []
    mi = []
    cluster_range = np.arange(18, 202)[::2]
    for n_cluster in cluster_range:
        clus_model.n_clusters = n_cluster
        clus_model.fit(to_be_fit)
        labels = clus_model.labels_

        nmi_score = normalized_mutual_info_score(true_label, labels)
        ari_score = adjusted_rand_score(true_label, labels)
        mi_score = mutual_info_score(true_label, labels)

        nmi.append(nmi_score)
        ari.append(ari_score)
        mi.append(mi_score)

    # Plot versus number of clusters
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].plot(cluster_range, nmi)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('NMI score')
    ax[0].set_title('NMI scores for varying number of clusters')

    ax[1].plot(cluster_range, ari)
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('ARI score')
    ax[1].set_title('ARI for varying number of clusters')

    ax[2].plot(cluster_range, mi)
    ax[2].set_xlabel('Number of clusters')
    ax[2].set_ylabel('MI score')
    ax[2].set_title('MI for varying number of clusters')

    plt.tight_layout()
    plt.savefig(filename)

    # Write txt report, change path name
    with open(filename.with_suffix('.txt'), "w") as f:
        f.write(f"NMI: {nmi}\n")
        f.write(f"ARI: {ari}\n")
        f.write(f"MI: {mi}\n")

    max_nmi = max(nmi)
    max_ari = max(ari)
    max_mi = max(mi)

    max_nmi_at = cluster_range[nmi.index(max_nmi)]
    max_ari_at = cluster_range[ari.index(max_ari)]
    max_mi_at = cluster_range[mi.index(max_mi)]

    return max_nmi, max_ari, max_mi, max_nmi_at, max_ari_at, max_mi_at
