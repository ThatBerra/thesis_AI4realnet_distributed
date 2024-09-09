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
Replace EpisodeData.py in `venv_clustering/lib/python3/site-packages/grid2op/Episode/EpisodeData.py`
Replace aux_fun.py in `venv_clustering/lib/python3/site-packages/grid2op/Runner/aux_fun.py`

#### Run experiment on synthetic data
```commandline
python clustering/synthetic_data.py
```
#### Run experiment on power grids
```commandline
python clustering/power_grid.py
```

