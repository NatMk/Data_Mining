from collections import Counter
import numpy as np


def get_neighbors(X, test_inst, k, labels):
    distances = []
    for i in range(len(X)):
        # Calculate distances from test point to each point of train set
        dist = np.linalg.norm(test_inst-X[i])
        distances.append((X[i], dist, labels[i]))
    distances.sort(key=lambda x: x[1])
    # Get k nearest train points to the datapoint
    neighbors = distances[:k]
    return neighbors

def myknn(X, test, k, labels):
    res = []
    # For each point in test set, assign label based on it's neighbors
    for i in range(len(test)):
        counter = Counter()
        neighbors = get_neighbors(X, test[i], k, labels)
        for n in neighbors:
            counter[n[2]] += 1
        res.append(counter.most_common(1)[0][0])
    return res

def accuracy_myknn(knn_labels, orig_labels):
    return sum([1 for i in range(len(knn_labels)) 
                if knn_labels[i]==orig_labels[i]])/len(knn_labels)

print("Performed k-nearest neighbor (k-NN) classification with n number of objects and p number of attributes."
      " X is training data, test is " + "testing data, and k is a user parameter.")