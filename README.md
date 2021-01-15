# Code Repository "Direct prediction of phonon density of states with Euclidean neural network"
This is the repository to accompany the paper. [https://arxiv.org/pdf/2009.05163.pdf](https://arxiv.org/pdf/2009.05163.pdf)

## Requirements

- `torch `
- `pymatgen`
- `torch-geometric`
- `e3nn=0.0.0`, installed from (https://github.com/e3nn/e3nn/releases/tag/0.0.0) 

Please notice the `e3nn` used for this work is not the latest version `0.1.1`, more details about versions of its dependencies used for this work can be found in `requirements.txt`. We will prepare another notebook for `e3nn=0.1.1` in near future.

## Notebooks and python files

`phdos_train.ipynb`
- Trains a E(3)NN model that predicts phonon density of states (DoS) based on crystal structures.

`phdos_eval_MPdata.ipynb` and `class_evaluate_MPdata.py`
- Evaluates phonon DoS and specific heat capacities for around 4,400 crystals on Materials Project.

`query_mpcifs.ipynb` and `data/mp_data.csv`
- Download `*.cif` files from the Materials Project, with material IDs listed in `data/mp_data.csv`.

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
