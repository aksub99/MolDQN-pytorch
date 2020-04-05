# MolDQN-pytorch
[![MIT
license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

PyTorch implementation of RetinaNet object detection as described in [Optimization of Molecules via Deep Reinforcement Learning](https://www.nature.com/articles/s41598-019-47148-x)
by Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N. Zare and Patrick Riley.

## Installation

## From source:

1) Install `rdkit`.  
   `conda create -c rdkit -n my-rdkit-env rdkit`  
   `conda activate my-rdkit-env`  
   `conda install -c conda-forge rdkit`  
   
2) Clone this repository.  
   `git clone https://github.com/aksub99/MolDQN-pytorch.git`.  
   
3) Install the requirements given in `requirements.txt`.  
   `pip install -r requirements.txt`  
   
4) Install `baselines`.  
   `pip install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"`  
   
