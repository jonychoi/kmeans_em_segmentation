# 2022 Computer Vision H.W #3
# All codes are written By Su Hyung Choi - 2018930740
# KMeans

import os
import numpy as np
from load_image import load_image_asarray
from numpy.random import uniform
from PIL import Image

# KU.raw (720x560 image, each pixel is an 8-bit number)
# Gundam.raw (600x600 image, each pixel is an 8-bit number)
# Golf.raw (800x540 image, each pixel is an 8-bit number)

def euclidean(point, data):
    # print("point: ", point)
    # print("data: ", data)
    # print("cost: ", np.sqrt(np.sum((point - data)**2, axis=1)))
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:
    def __init__(self, n_clusters, iter):
        self.n_clusters = n_clusters
        self.iter = iter
        self.out = None

    def fitting(self, x):
        min_, max_ = np.min(x, axis=0), np.max(x, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        iteration = 0
        prev_centroids = None
        
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.iter:
        
            sorted_points = [[] for _ in range(self.n_clusters)]
            idx_of_sorted = [[] for _ in range(self.n_clusters)]
            iter_cost = 0


            for i, _x in enumerate(x):
                cost = euclidean(_x, self.centroids)
                iter_cost += cost
                centroid_index = np.argmin(cost)
                sorted_points[centroid_index].append(_x)
                idx_of_sorted[centroid_index].append(i)

            # updating the centroids
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis = 0) for cluster in sorted_points]

            for i, centroid in enumerate(self.centroids):
                # Catch any np.nans, resulting from a centroid having no points
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
            print('{} iteration cost is: {}'.format(iteration, iter_cost))
        
        #make average grey after iteration complete
        for i, points in enumerate(sorted_points):
            avg = np.average(points)
            sorted_points[i] = np.full(len(points), avg)

            # print(i, "##########################")
            # print(sorted_points[i])

        for j, positions in enumerate(idx_of_sorted):
            for i, position in enumerate(positions):
                x[position] = sorted_points[j][i]

        self.out = x

    def output(self):
        return self.out
            

def main(filename, shape, n_clusters, iter):
    arr = load_image_asarray(filename = "./source/{}.raw".format(filename), shape = shape)
    
    kmeans = KMeans(n_clusters = n_clusters, iter = iter)
    print("{}'s {} cluster with {} iteration fitting started".format(filename, n_clusters, iter))
    kmeans.fitting(arr)

    output = kmeans.output().reshape(shape)

    save_dir = os.getcwd() + '/results/kmeans/'
    os.makedirs(save_dir, exist_ok= True)

    img = Image.fromarray(output)
    img.save(save_dir + "{}'s_{}_cluster_of_{}_iter.bmp".format(filename, n_clusters, iter))
    img.show()

main("Golf", (800, 540), 2, 10)
main("Golf", (800, 540), 4, 10)
main("Golf", (800, 540), 8, 10)

main("Gundam", (600, 600), 2, 10)
main("Gundam", (600, 600), 4, 10)
main("Gundam", (600, 600), 8, 10)

main("KU", (720, 560), 2, 10)
main("KU", (720, 560), 4, 10)
main("KU", (720, 560), 8, 10)