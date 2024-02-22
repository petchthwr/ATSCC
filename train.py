from atscc import *
import wandb

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Variable for wandb logging
wandb.login(key='dd48a2fe503f34d4c795ab3877f4fb93132843e1')
dataset = ['RKSIa_v']

split_point = 0.5
polar = True
direction = True

input_dims = 9
output_dims = 320
num_layers = 12
embed_dims = 768
ffn_dims = 3072
num_heads = 12

num_epochs = 5
eval_every = 5

sweep_config = {'method': 'grid'}
metric = {'name': 'NMI', 'goal': 'maximize'}
sweep_config['metric'] = metric
parameters_dict = {
    'seed': {
        'values': [0, 1, 2, 3, 4]
        },
    'rdp_epsilon': {
        'values': [0.0001]
        },
    'batch_size': {
        'values': [8]
        },
    'temperature': {
        'values': [0.05]
        },
    'dropout': {
        'values': [0.3]
        },
    'lr': {
        'values': [1e-5]
        }
    }
sweep_config['parameters'] = parameters_dict


def run_sweep_for_dataset(dset_name, sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="ATSCC_" + dset_name)

    def sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config

            reproducibility(config.seed)
            train_loader, test_loader = load_data(dset_name, split_point, downsample=10, size_lim=5000, rdp_epsilon=config.rdp_epsilon, batch_size=config.batch_size, device=device, polar=polar, direction=direction)
            Encoder = TSGPTEncoder(input_dims, output_dims, embed_dims, num_heads, num_layers, ffn_dims, config.dropout).to(device)
            optim = torch.optim.Adam(Encoder.parameters(), lr=config.lr, weight_decay=1e-5)
            loss_log, score_log = fit(Encoder, train_loader, test_loader, optim, num_epochs, eval_every, config.temperature, device, dset_name, verbose=False, visualize=False, pooling='last')

            for loss in loss_log:
                wandb.log({'loss': loss})
            for score_dict in score_log:
                wandb.log(score_dict)

    wandb.agent(sweep_id, function=sweep, count=50)


if __name__ == "__main__":
    for dset_name in dataset:
        run_sweep_for_dataset(dset_name, sweep_config)