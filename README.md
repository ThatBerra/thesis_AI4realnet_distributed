# thesis_AI4realnet_distributed

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
Replace `grid2op_patch/EpisodeData.py` in `venv_clustering/lib/python3/site-packages/grid2op/Episode/EpisodeData.py`

Replace `grid2op_patch/aux_fun.py` in `venv_clustering/lib/python3/site-packages/grid2op/Runner/aux_fun.py`

#### Run experiment on synthetic data
```commandline
python clustering/synthetic_data.py
```
#### Run experiment on power grids
```commandline
python clustering/power_grid.py
```
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
#### Run experimenta 
```commandline
python distributed_grid2op/main.py
```


