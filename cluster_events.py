import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

def determine_optimal_k(X, K): # kmin=10, kmax=50,
    loss = []
    # K = range(kmin, kmax)
    for k in K:
        km = MiniBatchKMeans(n_clusters=k)
        clusters = km.fit_predict(X)
        cur_loss = sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
        loss.append(cur_loss)
    return loss

def get_clusters(events_df, k):
    event_ids = events_df[['event_id']]
    event_features = events_df.iloc[:, 9:] #the last 101 cols
    km = MiniBatchKMeans(n_clusters=k)
    clusters = pd.DataFrame(km.fit_predict(event_features))

    event_clusters = pd.concat([event_ids, clusters], axis=1, join='inner')
    event_clusters.columns = ['event_id', 'cluster']

    centers = km.cluster_centers_
    ids = clusters.to_numpy().reshape(-1)
    event_centroids = centers[ids]
    event_clusters['centroid'] = event_centroids.tolist()

    return event_clusters