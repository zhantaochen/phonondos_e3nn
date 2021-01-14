# Code Repository "Direct prediction of phonon density of states with Euclidean neural network"
This is the repository to accompany the paper. [https://arxiv.org/pdf/2009.05163.pdf](https://arxiv.org/pdf/2009.05163.pdf)

## Requirements

- `torch `
- `pymatgen`
- `torch-geometric`
- `e3nn`

## Notebooks and python files

`phdos_train.ipynb`
- Trains a E(3)NN model that predicts phonon density of states (DoS) based on crystal structures.

`phdos_eval_MPdata.ipynb` and `class_evaluate_MPdata.py`
- Evaluates phonon DoS and specific heat capacities for around 4,400 crystals on Materials Project.

`query_mpcifs.ipynb`
- Download `*.cif` files from the Materials Project.

## Citing

```
@article{chen2020direct,
  title={Direct prediction of phonon density of states with Euclidean neural network},
  author={Chen, Zhantao and Andrejevic, Nina and Smidt, Tess and Ding, Zhiwei and Chi, Yen-Ting and Nguyen, Quynh T and Alatas, Ahmet and Kong, Jing and Li, Mingda},
  journal={arXiv preprint arXiv:2009.05163},
  year={2020}
}
```
