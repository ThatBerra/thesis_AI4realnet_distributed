#Copyright Weihao Gao, UIUC

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np

#Main Function
def Mixed_KSG(x,y,k=5):
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])   	
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis, _ = tree_xy.query(data, [k+1], p=float('inf'))
	knn_dis = np.squeeze(knn_dis) 
	ans = 0

	for i in range(N):
		kp, nx, ny = k, k, k
		if knn_dis[i] == 0:
			kp = len(tree_xy.query_ball_point(data[i],1e-20,p=float('inf')))
			nx = len(tree_x.query_ball_point(x[i],1e-20,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],1e-20,p=float('inf')))
		else:
			nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-20,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-20,p=float('inf')))
		ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
	return ans



