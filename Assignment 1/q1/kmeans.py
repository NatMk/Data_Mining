from copy import deepcopy
import numpy as np


def mykmeans(X, k):
    # Choose k random points from the dataset as inital centroids
    centroids = X[np.random.randint(len(X), size=k)]
    # Array for storing previous centroids
    centroids_old = np.zeros(centroids.shape)
    # Array for clusters
    clusters = np.zeros(len(X), dtype=np.int8)
    # Loop untill centers stop changing
    while np.linalg.norm(centroids - centroids_old) != 0:
        for i in range(len(X)):
            # calculate distances between the datapoint and centroids
            distances = np.linalg.norm(X[i]-centroids, axis=1)
            # Asign datapoint to cluster
            clusters[i] = np.argmin(distances)
        # Save previous centroids
        centroids_old = deepcopy(centroids)
        # Update centroids
        for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            centroids[i] = np.mean(points, axis=0)
    return clusters, centroids

print("Clustered n number of objects and p number of attributes into k clusters.")