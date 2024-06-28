# ATSCC: Aircraft Trajectory Segmentation-based Contrastive Coding

![Screenshot](atscc.png)

ATSCC is a self-supervised time series representation for multivariate aircraft trajectory. It allows for the semantic analysis of air traffic trajectory data. This repository contains the official implementation for the ATSCC framework as described in the paper titled "Aircraft Trajectory Segmentation-based Contrastive Coding: A Framework for Self-supervised Trajectory Representation."

## Requirements

The recommended requirements for ATSCC are listed as follows:
* libopenvino-pytorch-frontend=2024.0.0
* matplotlib=3.8.2
* matplotlib-base=3.8.3
* matplotlib-inline=0.1.6
* numpy=1.26.4
* pandas=2.1.4
* rdp=0.8
* scipy=1.12.0
* seaborn=0.13.2
* seaborn-base=0.13.2
* torch=2.0.1+cu118
* torchaudio=2.0.2+cu118
* torchvision=0.15.2+cu118
* tsaug=0.2.1
* umap-learn=0.5.4
* wandb=0.16.4

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Dataset
The code implementation utilizes the ATFMTraj data accessible via [ATFMTraj](https://huggingface.co/datasets/petchthwr/ATFMTraj). The dataset comprises 3 airports, totaling 4 datasets.
- **Incheon International Airport (RKSI)**: Data from 2018 to 2023 were obtained from the Opensky database, focusing on flights identified via Airportal schedules. The arrival and departure datasets are labeled RKSIa and RKSId, respectively.
- **Stockholm Arlanda Airport (ESSA)**: This implementation utilizes data from the Swedish Civil Air Traffic Control (SCAT) dataset, which includes surveillance, weather, and flight plans. It primarily examines arrival data.
- **Zurich Airport (LSZH)**: The analysis focuses on arrival flight trajectories using a dataset referenced in the study, emphasizing the evaluation of this data for our experiments.


## Example

```python
from atscc import *
from model.GPT import TSGPTEncoder
from sklearn.cluster import KMeans
from ATFMTraj import load_ATFM

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

train_data, train_labels = load_ATFM('RKSIa_v', 'TRAIN', path_to_data)
test_data, test_labels = load_ATFM('RKSIa_v', 'TEST', path_to_data)

train_loader, test_loader = load_data(dataset='RKSIa_v',
                                      split_point='auto',
                                      downsample=5,
                                      size_lim=None,
                                      rdp_epsilon=0.1,
                                      batch_size=16,
                                      device=device,
                                      polar=True,
                                      direction=True,
                                      x = (train_data, train_labels, test_data, test_labels))

Encoder = TSGPTEncoder(input_dims=9,
                       output_dims=320,
                       embed_dim=768,
                       num_heads=12,
                       num_layers=12,
                       hidden_dim=3072,
                       dropout=0.35).to(device)

loss_log, score_log = fit(Encoder, train_loader, test_loader,
                          optim=torch.optim.AdamW(Encoder.parameters(), lr=1e-5, weight_decay=1e-5),
                          scheduler=None,
                          num_epochs=10,
                          max_iter=500000,
                          eval_every=5,
                          local_temp=5.0,
                          device=device,
                          data='RKSIa_v',
                          clus_model=KMeans(n_clusters=2))
```

## Example embedding

![Screenshot](embedding.png)

## Citation
```bibtex
@dataset{ATFMTraj2024,
  title={ATFMTraj: Aircraft Trajectory Classification Data for Air Traffic Management},
  author={Phisannupawong, Thaweerath and Damanik, Joshua Julian and Choi, Han-Lim},
  year={2024},
  note={https://huggingface.co/datasets/petchthwr/ATFMTraj}
}

@misc{ATSCC2024,
  title={Aircraft Trajectory Segmentation-based Contrastive Coding: A Framework for Self-supervised Trajectory Representation},
  author={Phisannupawong, Thaweewrath and Damanik, Joshua J. and Choi, Han-Lim},
  year={2024},
  note={Preprint submitted for publication}
}
```
