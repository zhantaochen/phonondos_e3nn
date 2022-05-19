# Code Repository for "Direct prediction of phonon density of states with Euclidean neural network"
This is the repository to accompany the paper: [https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214](https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214) (or on arXiv [https://arxiv.org/pdf/2009.05163.pdf](https://arxiv.org/pdf/2009.05163.pdf)). Please direct any questions to Zhantao (zhantao@mit.edu).

We later rewrote the code with latest `e3nn` in [https://github.com/ninarina12/phononDoS_tutorial](https://github.com/ninarina12/phononDoS_tutorial), which was prepared for the [E3NN tutorial in MRS 2021 Fall Meeting](http://e3nn.org/e3nn-tutorial-mrs-fall-2021/#tut6).

## Dependency installation instructions

These are very explicit installation instructions that do NOT assume you have experience installing specific python packages or using virtual environments. It's important that you follow these detailed instructions because one of our important dependencies `torch-geometric` uses custom CUDA kernels (for Nvidia GPUs) that are compiled for specific combinations of `torch` and CUDA (e.g. version 9.2, 10.1, or 10.2). These instructions assume you have `python 3.6+` installed.

We require the following specific package verisons:
- `torch==1.5.1`
- `torch-geometric==1.5.0`
- `e3nn` at commit [`4653db57d0af05e1834e65b6da73fa26927824bc`](https://github.com/e3nn/e3nn/tree/4653db57d0af05e1834e65b6da73fa26927824bc)

Please see below for how to install these packages.

### Clone this repository

To make a copy of this repository on your computer run the following at the command line.  
```git clone https://github.com/zhantaochen/phonondos_e3nn.git```

The change into the directory.  
```cd phonodos_e3nn```

### Setting up a virtual environment

Because we use specific versions of these packages, you may want to make and activate a new virtual environment. In the command line run,  
```
virtualenv --python=`which python3` venv
source venv/bin/activate
```
Whenver you want to exit this environment (not right now because we want to install packages), enter `deactivate` in the command line. You can reactivate the environment using `source venv/bin/activate` like above.

### Installing dependencies

First, let's install torch and torchvision.  
```pip install torch==1.5.1 torchvision==0.6.1```

Next we will install `torch-geometric` which is a versatile library for doing deep learning on graphs. Please first check the following **before** installing `torch-geometric`.

In a python, check the CUDA version `torch` is using.  
```
import torch
print(torch.version.cuda)
```

At the command line, check your NVIDIA compiler version.  
```nvcc --version```

If you do not have the `nvcc` command, this likely means you do not have CUDA installed and / or do not have a GPU. If this is the case, you will set `CUDA=cpu` in the be

These versions **must** match before proceeding. If they match, at the command line set the variable `CUDA` to be `cu92`, `cu101`, or `cu102` to match your CUDA version or `cpu` to not install GPU dependences. For example, `CUDA=cu102` will set the CUDA variable to indicate you want to install a version for CUDA 10.2. `CUDA=cpu` will install torch-geometric without GPU dependencies. You can check that the variable `CUDA` is set by running `echo CUDA`.

Then run the following.  
```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install torch-geometric==1.5.0
```

Then install `e3nn` at commit [`4653db57d0af05e1834e65b6da73fa26927824bc`](https://github.com/e3nn/e3nn/tree/4653db57d0af05e1834e65b6da73fa26927824bc).  
```pip install git+git://github.com/e3nn/e3nn.git@4653db57d0af05e1834e65b6da73fa26927824bc```

and finally `pymatgen` which we use to interact with the Materials Project API and manipulate crystal structure data. Installing `e3nn` will likely have already installed this package, but you can also install it with.  
```pip install pymatgen==2020.6.8```

### Using a virtual environment with Jupyter
In order to use this virtual environment in Jupyter you will need to add it to with `ipykernel`.  
```
pip install ipykernel
python -m ipykernel install --user --name=phonondos_e3nn
```

Now, if you start a `jupyter notebook` outside this virtual enviroment (like in a new command line window, but still in the `phonondos_e3nn` folder) and navigate to wherever these notebook are hosted, for example `localhost:8888`, you will be able to select the `phonondos_e3nn` Kernel from the Kernel drop down menu in Jupyter when you view any of the notebooks in this repository.

### Download data from the Materials Project
This repository uses data from the Materials Project. To access the Materials Project dataset via pymatgen, you will need an API key which can obtain via https://www.materialsproject.org/dashboard. See https://pymatgen.org/pymatgen.ext.matproj.html for more details.

One you have your Materials Project API key, you can add it to your pymatgen config file (~/.pmgrc.yaml) with the following command.  
```pmg config --add PMG_MAPI_KEY YOUR_API_KEY```

To download the data needed, run the following command  
```python download_mpcifs.py```  
or to set the API key manually.  
```python download_mpcifs.py YOUR_API_KEY```

### Other details
These dependencies (also called requirements) can be found in `requirements.txt` but **installing with `pip install -r requirements.txt` will NOT work so please follow the instructions above**. If you run into problems installing `pytorch` or `torch-geometric`, please consult the official [pytorch](https://pytorch.org/get-started/previous-versions/) and [torch-geometric](https://github.com/rusty1s/pytorch_geometric#installation) installation instructions for relevant information.

This repository uses a specific commit of `e3nn` between `0.0.0` and `0.0.1` releases (commit [`4653db57d0af05e1834e65b6da73fa26927824bc`](https://github.com/e3nn/e3nn/tree/4653db57d0af05e1834e65b6da73fa26927824bc)); you **must** use this version to ensure that you can import saved model parameters. If your `torch-geometric`  version is equal or higher than `1.6.0`, then additional fixes from the next commit are needed, please [check commits history here.](https://github.com/e3nn/e3nn/commits/master?after=447ccb253061a50b29f3a05c6eeffba34cca2c14+174&branch=master)

We are planning to prepare another notebook for the latest release of `e3nn` and dependencies in near future.

## Notebooks and python files
`download_mpcifs.py` and `data/mp_data.csv`
- Download `*.cif` files from the Materials Project, with material IDs listed in `data/mp_data.csv`.
- Usage: `python download_mpcifs.py YOUR_MAPI_KEY`

`phdos_train.ipynb`
- Trains a E(3)NN model that predicts phonon density of states (DoS) based on crystal structures.

`phdos_eval_MPdata.ipynb` and `class_evaluate_MPdata.py`
- Evaluates phonon DoS and specific heat capacities for around 4,400 crystals on Materials Project.

`models/200803-1018_len51max1000_fwin101ord3_trial_run_full_data.torch`
- The trained model presented in the paper.

`models/cif_unique_files.pkl`
- Contains Materials Project IDs that used for searching high heat capacity materials, the materials presented in traning/validation/testing datasets are excluded.

`models/phdos_e3nn_len51max1000_fwin101ord3.zip`
- Zipped file that contains interpolated phonon DoS and other necessary information for training (cif files, etc.). The included information is curated from the dataset presented in the work: [Petretto, Guido, et al. "High-throughput density-functional perturbation theory phonons for inorganic materials." Scientific data 5 (2018): 180065.](https://www.nature.com/articles/sdata201865)

## Citing

```
@article{chen2020direct,
  title={Direct prediction of phonon density of states with Euclidean neural network},
  author={Chen, Zhantao and Andrejevic, Nina and Smidt, Tess and Ding, Zhiwei and Chi, Yen-Ting and Nguyen, Quynh T and Alatas, Ahmet and Kong, Jing and Li, Mingda},
  journal={arXiv preprint arXiv:2009.05163},
  year={2020}
}
```
