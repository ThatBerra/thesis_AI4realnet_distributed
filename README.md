# Distributed Reinforcement Learning for Power Grid Operations
The code on this repository is used to for the research on distributed RL applied to power grid operations. The code is used to run the experiments presented in the Master Thesis [link] carried on at Politecnico di Milano. 

The underlying idea is to decompose the problem by splitting the MDP into sub-problems by estimating Mutual Information between pairs of state and action variables, that are then clustered so that variables that have high correlation with the same ones are grouped together. The code for this part is collected in the `clustering` folder. We made an experiment on a custom made MDP and one on a simple Grid2op environment.

The code in the folder `distributed_grid2op` develops a distributed RL approach in which each sub-problem is handled independently by a dedicated agent.

Below, the instruction for a proper execution.

## Clustering

### Setup virtual env
#### Create a virtual environment
```commandline
cd thesis_AI4realnet_distributed
pip3 install -U virtualenv
python3 -m virtualenv venv_clustering
```
#### Activate virtual environment
```commandline
source venv_clustering/bin/activate
```
#### Install required packages
```commandline
pip install -r requirements_clust.txt
```
Replace `grid2op_patch/EpisodeData.py` in `.../venv_clustering/lib/python3.11/site-packages/grid2op/Episode/EpisodeData.py`

Replace `grid2op_patch/aux_fun.py` in `.../venv_clustering/lib/python3.11/site-packages/grid2op/Runner/aux_fun.py`

### Run experiments

#### Run experiment on synthetic data
```commandline
python clustering/synthetic_data.py
```
#### Run experiment on power grids
```commandline
python clustering/power_grid.py
```
One can modify the two variables `n_episodes` and `n_samples`. The first one is the number of time series that are used in the simulation for collecting the data, the second is the number of samples that are used in the computation of the Mutual Information estimator. The total number of samples collected depends on the survival of the agent in the simulation, if it lower than `n_samples`, the MI is computed on all available samples. 

#### Reproducing thesis experiment
To reproduce the results provided in the thesis, use the following settings:
```commandline
SEED = 29
n_episodes = 1000
n_samples = 50000
```
Tested on Ubuntu 18.04.6 LTS | RAM 8GB | Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 
Running time: ~1h on 6 cores

#### Disclaimer
Creating a virtual environment is not strictly necessary. However, due to the fact that in order to properly run the experiment on power grid one needs to modify files in the library, this is strongly suggested, to be sure that grid2op works as intended outside the virtual environment. It is also suggested to install packages from requirements_clust.py, as later versions of grid2op generate conflicts with the two files `aux_fun.py` and `EpisodeData.py`

## Learning
### Setup virtual env
#### Create a virtual environment
```commandline
cd thesis_AI4realnet_distributed
pip3 install -U virtualenv
python3 -m virtualenv venv_learning
```
#### Activate virtual environment
```commandline
source venv_learning/bin/activate
```
#### Install required packages
```commandline
pip install -r requirements.txt
```
### Run experiments
```commandline
python distributed_grid2op/main.py
```
#### Reproducing thesis experiment
To reproduce the results provided in the thesis, use the following settings:
```commandline
SEED = 90566
```
`n_iterations` value depends on how much one wants to train its model, we used 100 for the parameter exploration and 25 for the comparison with centralized models.


