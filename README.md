# AV-SepFormer

This Git repository for the official PyTorch implementation of ""**"[AV-SepFormer: Cross-attention SepFormer for Audio-Visual Target Speaker Extraction](https://ieeexplore.ieee.org/document/10094306)"**, accepted by ICASSP 2023.

📜[[Full Paper](https://ieeexplore.ieee.org/document/10094306)] ▶[[Demo](https://lin9x.github.io/AV-SepFormer_demo/)] 💿[[Checkpoint](https://drive.google.com/drive/folders/1M26x5qCE0LuaEtZ76E_YSnW0hf_chGeq?usp=drive_link)]              


## Requirements

  - Linux
  
  - python >= 3.8

  - Anaconda or Miniconda

  - NVIDIA GPU + CUDA CuDNN (CPU can also be supported)


## Environment && Installation

Install Anaconda or Miniconda, and then install conda and pip packages:

```shell
# Create conda environment
conda create --name av_sep python=3.8
conda activate av_sep

# Install required packages
pip install -r requiremens.txt
```

## Start Up
Clone the repository:

```shell
git clone https://github.com/lin9x/AV-Sepformer.git
cd AV-Sepformer
```

### Data preparation
Scripts to preprocess the voxceleb2 datasets is the same as which in MuSE. You can dirctly go to this [repository](https://github.com/zexupan/MuSE) to preprocess your data.
Pairs of our data is in data_list
*
## Training
First, you need to modify the various configurations in config/avsepformer.yaml for training.

Then you can run training:

```shell
source activate av_sep
CUDA_VISIBILE_DEVISIBLE=0,1 python3 run_avsepformer.py run config/avsepformer.yaml
```

If you want to train other audio-visual speech separation systems, **AV-ConvTasNet** and **MuSE** is available in our repo. Just turn to the run_**system**.py and config/**system**.yaml to train your own model.


# References
The data preparation follows the operation in [MuSE](https://arxiv.org/abs/2010.07775) Github Repository.
