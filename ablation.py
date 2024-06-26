import torch.optim
from atscc import *
import wandb
from model.GPT import TSGPTEncoder
from model.encoder import TSEncoder, TSEncoderLSTM
from sklearn.cluster import KMeans

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Variable for wandb logging
wandb.login(key='dd48a2fe503f34d4c795ab3877f4fb93132843e1')

split_point = 'auto'
direction = False
polar = True

input_dims = 6
output_dims = 320
num_layers = 12
embed_dims = 768
ffn_dims = 3072
num_heads = 12

dropout = 0.35
batch_size = 16
lr = 1e-5

num_epochs = 10
eval_every = 5
max_iter = 500000

best_params = { # This will not be used for sweep
    'rdp_epsilon': {'RKSIa_v': 0.0001, 'RKSId_v': 0.1, 'ESSA_v': 0.0001, 'LSZH_v': 0.01},
    'temperature': {'RKSIa_v': 10.0, 'RKSId_v': 5.0, 'ESSA_v': 0.01, 'LSZH_v': 0.1}
}

sweep_config = {'method': 'grid'}
metric = {'name': 'NMI', 'goal': 'maximize'}
sweep_config['metric'] = metric
parameters_dict = {
    'seed': {
        'values': [0, 1, 2, 3, 4]
        },
    'dset_name': {
        'values': ['RKSIa_v']
        },
    'tag': {
        'values': ['w/o direction']
        },
    }
sweep_config['parameters'] = parameters_dict

def run_sweep_for_dataset(sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="ATSCC_Ablation")

    def sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config

            reproducibility(config.seed)
            train_loader, test_loader = load_data(config.dset_name, split_point, downsample=5, size_lim=None, rdp_epsilon=best_params['rdp_epsilon'][config.dset_name],
                                                  batch_size=batch_size, device=device, polar=polar, direction=direction)

            #Encoder  = TSEncoder(input_dims, output_dims).to(device)
            #Encoder = TSEncoderLSTM(input_dims, output_dims).to(device)
            Encoder = TSGPTEncoder(input_dims, output_dims, embed_dims, num_heads, num_layers, ffn_dims, dropout, ablationtag=config.tag).to(device)
            optim = torch.optim.AdamW(Encoder.parameters(), lr=lr, weight_decay=1e-5)
            clus_model = KMeans(n_clusters=2, random_state=config.seed)

            loss_log, score_log = fit(Encoder, train_loader, test_loader, optim, None, num_epochs, max_iter, eval_every,
                                      best_params['temperature'][config.dset_name], device, config.dset_name, clus_model,
                                      verbose=False, visualize=True, pooling='last', eval_method='clustering')

            # Encode
            ori_train_loader_eval = train_loader.dataset.eval
            ori_train_collate_fn = train_loader.collate_fn
            train_loader.dataset.eval = True
            train_loader.collate_fn = pad_stack_test
            train_repr, train_traj, train_full_encode, split_ind_train, train_label = encode(Encoder, train_loader, device, pooling='last')
            test_repr, test_traj, test_full_encode, split_ind_test, test_label = encode(Encoder, test_loader, device, pooling='last')
            train_loader.dataset.eval = ori_train_loader_eval
            train_loader.collate_fn = ori_train_collate_fn

            np.save(f'repr/{config.dset_name}_ATSCC_seed{config.seed}.npy', test_repr)
            np.save(f'repr/{config.dset_name}_ATSCC_seed{config.seed}_train.npy', train_repr)

            # Save full encode
            np.save(f'repr/{config.dset_name}_ATSCC_fullencode_seed{config.seed}.npy', test_full_encode)
            np.save(f'repr/{config.dset_name}_ATSCC_fullencode_seed{config.seed}_train.npy', train_full_encode)

            for loss in loss_log:
                wandb.log({'loss': loss})
            for score_dict in score_log:
                wandb.log(score_dict)

    wandb.agent(sweep_id, function=sweep, count=50)


if __name__ == "__main__":
    run_sweep_for_dataset(sweep_config)