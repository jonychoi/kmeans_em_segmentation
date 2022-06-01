import numpy as np
from load_image import load_image
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns

# KU.raw (720x560 image, each pixel is an 8-bit number)
# Gundam.raw (600x600 image, each pixel is an 8-bit number)
# Golf.raw (800x540 image, each pixel is an 8-bit number)

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    def fit(self, X_train):
        # Randomly select centroid start points, uniformly distributed across the domain of the dataset
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            print("##################################")
            print(self.centroids)
            print("##################################")
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idx


# X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
# X_train = StandardScaler().fit_transform(X_train)

def plot2d(X_train, centers, true_labels = None):
    # Create a dataset of 2D distributions

    # Fit centroids to dataset
    # print(X_train.shape, true_labels.shape)
    kmeans = KMeans(n_clusters=centers)
    kmeans.fit(X_train)
    
    # View results
    # centroids, centroid_idx = kmeans.evaluate(X_train)

    # print(centroids, centroid_idx)
    # print(classification)
    # # sns.scatterplot(x=[X[0] for X in X_train],
    # #                 y=[X[1] for X in X_train],
    # #                 hue=true_labels,
    # #                 style=classification,
    # #                 palette="deep",
    # #                 legend=None
    # #                 )
    # sns.scatterplot(x=[X[0] for X in X_train],
    #                 y=[X[1] for X in X_train],
    #                 style=classification,
    #                 palette="deep",
    #                 legend=None
    #                 )
    # plt.plot([x for x, _ in kmeans.centroids], [y for _, y in kmeans.centroids], 'k+', markersize=10)
    # plt.savefig("gundam.png")
    # plt.show()

X_train = load_image('./source/KU.raw', (-1, 2), to_array=True)
# print(X_train.shape)
plot2d(X_train, 2)



# def kmeans(filename, iteration = None):
#     arr = load_image(filename, (720, 560), to_array=True)
#     print(arr);
#     kmeans = KMeans(n_clusters=560, random_state=0).fit(arr)
#     print(kmeans.labels_)
#     #print(kmeans.predict([[0, 0], [12, 3]]))
#     print(kmeans.cluster_centers_)


# kmeans('./source/KU.raw')