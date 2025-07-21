import numpy as np
from sklearn.cluster import KMeans

def assign_kmeans_clusters(df, embedding_col='scaled_embedding', n_clusters=5, cluster_col='cluster'):
    embeddings = np.vstack(df[embedding_col].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    df[cluster_col] = cluster_labels
    return df

def add_outlier_flag(df, embedding_col='scaled_embedding', cluster_col='cluster', outlier_col='is_outlier', threshold_std=2.0):
    embeddings = np.vstack(df[embedding_col].values)
    clusters = df[cluster_col].values
    n_clusters = len(np.unique(clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    distances = np.linalg.norm(embeddings - centroids[clusters], axis=1)
    outlier_flags = np.zeros(len(df), dtype=bool)

    for c in range(n_clusters):
        cluster_distances = distances[clusters == c]
        mean_dist = cluster_distances.mean()
        std_dist = cluster_distances.std()
        is_outlier = (distances > mean_dist + threshold_std * std_dist) & (clusters == c)
        outlier_flags = outlier_flags | is_outlier

    df[outlier_col] = outlier_flags
    return df