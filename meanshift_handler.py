from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance
import numpy as np


def get_best_meanshift_model(X):
    # TODO quantile hyperparameter can be changed
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=X.shape[0])
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    return ms

def get_cluster(state_vector, model):
    center_idx = model.predict([state_vector])[0]
    return model.cluster_centers_[center_idx]
