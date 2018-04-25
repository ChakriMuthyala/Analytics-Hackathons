"""
Created on Mon Apr 17 23:21:59 2018

@author: CHAKRI
"""

import numpy as np
import sys
from copy import deepcopy


#------------Text Preprocessing-------------
k_cluster = int(sys.argv[-1])
file = open(sys.argv[-2], 'r')
np_conv = [[float(s) for s in t.split()] for t in file.readlines()]
np_conv = np.array(np_conv, dtype=np.float)

if k_cluster > np_conv.shape[0]:
    print('No.of clusters should be less than or equal to no.of d-dimensional points')
    sys.exit()

def random_k_centroids():
    centroids = [np.random.uniform(1.1*np.min(np_conv), 0.9*np.max(np_conv), size=np_conv.shape[1])
            for i in range(k_cluster)]
    centroids = np.array(centroids)
    centroids = np.array(centroids, dtype=np.float)
    centroids_prev = np.zeros(centroids.shape)  
    d_centroids = np.linalg.norm(centroids - centroids_prev)
    return centroids, centroids_prev, d_centroids




#z = np.array(list(zip(f1, f3)))
#Since these clusters points locates near to centroid 
#I assumed a upper bound i.e less than 10% of max value
#and lower bound greater than 10% of min value as initial assumption

#------Creating k_cluster random centroids for D-dimensioanl data
C, C_prev, C_dist_error = random_k_centroids()
clusters = np.zeros(len(np_conv))


#-----K Means clustering implementation with all required parameters
while C_dist_error != 0:
    
    for j in range(len(np_conv)):
        distances = [np.linalg.norm(np_conv[j]-C[i]) for i in range(k_cluster)]
        cluster = np.argmin(distances)
        clusters[j] = cluster
        #print(distances)
        if (np.isnan(distances).any()):
            C, C_prev, C_dist_error  = random_k_centroids()            

    C_prev = deepcopy(C)
    for i in range(k_cluster):
        points = [np_conv[j]  for j in range(len(np_conv)) if clusters[j] == i ]
        C[i] = np.mean(points, axis=0)
    C_dist_error = np.linalg.norm(C - C_prev)
    
    
np.savetxt('clusters.txt', C, fmt='%.2f')
#---Best view in sublime text editor or any editor other than microsoft's
    

