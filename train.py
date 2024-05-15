import torch.optim

from atscc import *
import wandb
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Variable for wandb logging
wandb.login(key='dd48a2fe503f34d4c795ab3877f4fb93132843e1')
dataset = ['RKSId_v']

split_point = 'auto'
polar = True
direction = True

input_dims = 9
output_dims = 320
num_layers = 12
embed_dims = 768
ffn_dims = 3072
num_heads = 12

num_epochs = 10
eval_every = 5
max_iter = 500000

sweep_config = {'method': 'grid'}
metric = {'name': 'NMI', 'goal': 'maximize'}
sweep_config['metric'] = metric
parameters_dict = {
    'seed': {
        'values': [0, 1, 2, 3, 4] # Fixed
        },
    'rdp_epsilon': {
        'values': [0.0001, 0.001, 0.01, 0.1] # RKSIa: 0.0001, RKSId: 0.1, ESSA: 0.0001, LSZH: 0.01
        },
    'batch_size': {
        'values': [16] # Fixed
        },
    'temperature': {
        'values': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0] # RKSIa: 10.0, RKSId: 5.0, ESSA: 0.01, LSZH: 0.1
        },
    'dropout': {
        'values': [0.35] # Fixed
        },
    'lr': {
        'values': [1e-5] # Fixed
        }
    }
sweep_config['parameters'] = parameters_dict

def run_sweep_for_dataset(dset_name, sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="ATSCC_" + dset_name)

    def sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config

            reproducibility(config.seed)
            train_loader, test_loader = load_data(dset_name, split_point, downsample=5, size_lim=None, rdp_epsilon=config.rdp_epsilon[dset_name],
                                                  batch_size=config.batch_size, device=device, polar=polar, direction=direction)
            Encoder = TSGPTEncoder(input_dims, output_dims, embed_dims, num_heads, num_layers, ffn_dims, config.dropout).to(device)
            optim = torch.optim.AdamW(Encoder.parameters(), lr=config.lr, weight_decay=1e-5)
            #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2500, gamma=0.5)
            clus_model = KMeans(n_clusters=2, random_state=config.seed)
            loss_log, score_log = fit(Encoder, train_loader, test_loader, optim, None, num_epochs, max_iter, eval_every, config.temperature[dset_name], device, dset_name, clus_model,
                                      verbose=False, visualize=False, pooling='last', eval_method='clustering')

            if len(parameters_dict['seed']) > 1:
                print('Saving representation for seed', config.seed)
                repr, _, _, _ = encode(Encoder, test_loader, device, pooling='last')
                np.save(f'repr/{dset_name}_ours_seed{config.seed}.npy', repr)

            for loss in loss_log:
                wandb.log({'loss': loss})
            for score_dict in score_log:
                wandb.log(score_dict)

    wandb.agent(sweep_id, function=sweep, count=50)


if __name__ == "__main__":
    for dset_name in dataset:
        run_sweep_for_dataset(dset_name, sweep_config)