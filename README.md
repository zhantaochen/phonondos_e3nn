# Code Repository "Direct prediction of phonon density of states with Euclidean neural network"
This is the repository to accompany the paper. [https://arxiv.org/pdf/2009.05163.pdf](https://arxiv.org/pdf/2009.05163.pdf)

## Requirements

- `torch `
- `pymatgen`
- `torch-geometric`
- `e3nn=0.0.0`, installed from (https://github.com/e3nn/e3nn/releases/tag/0.0.0) 

Please notice the package version used for this work is not the latest one. We will prepare another notebook for `e3nn=0.1.1` in near future.

## Notebooks and python files

`phdos_train.ipynb`
- Trains a E(3)NN model that predicts phonon density of states (DoS) based on crystal structures.

`phdos_eval_MPdata.ipynb` and `class_evaluate_MPdata.py`
- Evaluates phonon DoS and specific heat capacities for around 4,400 crystals on Materials Project.

`query_mpcifs.ipynb`
- Download `*.cif` files from the Materials Project, with material IDs listed in `data/mp_data.csv`.

## Citing

```
@article{chen2020direct,
  title={Direct prediction of phonon density of states with Euclidean neural network},
  author={Chen, Zhantao and Andrejevic, Nina and Smidt, Tess and Ding, Zhiwei and Chi, Yen-Ting and Nguyen, Quynh T and Alatas, Ahmet and Kong, Jing and Li, Mingda},
  journal={arXiv preprint arXiv:2009.05163},
  year={2020}
}
```
