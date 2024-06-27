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

## Example embedding

![Screenshot](embedding.png)

## Reference
