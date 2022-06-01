# 2022 Computer Vision H.W #3
# All codes are written By Su Hyung Choi - 2018930740
# EM

import numpy as np
from load_image import load_image_asarray
from numpy.random import uniform
from PIL import Image
import os

# KU.raw (720x560 image, each pixel is an 8-bit number)
# Gundam.raw (600x600 image, each pixel is an 8-bit number)
# Golf.raw (800x540 image, each pixel is an 8-bit number)


def cost_reconstruction(x, means_of_data):
    diff = x-means_of_data
    error = np.sqrt(np.sum(np.power(diff, 2)))
    return error


def init_parameters(x, n_clusters):
    print(n_clusters)
    mean = np.random.rand(n_clusters)
    min_, max_ = np.min(x, axis=0), np.max(x, axis=0)
    var = [uniform(min_, max_) for _ in range(n_clusters)]
    pi = np.full(n_clusters, 1 / n_clusters)
    return mean, var, pi

def f_logsumexp(x, mean, var, pi):
    N = x.shape[0]
    K = mean.shape[0]
    trick = np.zeros((N, K))
    for k in range(K):
        subtraction = np.subtract(x, mean[k])
        arg1 = -1.0 / (2 * var[k]) * np.power(subtraction, 2)
        arg2 = -0.5*np.log(2*np.pi*var[k])
        arg3 = np.sum(arg2 + arg1, axis=1)  # before sum 1xD -> 1x1
        arithmitis = np.log(pi[k]) + arg3
        trick[:, k] = arithmitis
    # find max of all fk(trick[k]) for each example
    m = trick.max(axis=1)  # Nx1
    m = m.reshape((m.shape[0], 1))
    return trick, m


def update_loglikehood(f, m):
    f = f - m
    arg1 = np.sum(np.exp(f), axis=1)
    arg1 = np.log(arg1)
    arg1 = arg1.reshape((arg1.shape[0], 1))
    arg2 = arg1+m
    return np.sum(arg2, axis=0)

# K -> number of clusters
def update_gamma(f, m):
    f = f-m
    f = np.exp(f)   # NxK
    par = np.sum(f, axis=1)  # Nx1
    par = par.reshape((par.shape[0],1))
    result = np.divide(f, par)   # NxK
    return result


# return matrix with dimensions KxD
def update_mean(gamma, x):
    arith = np.dot(np.transpose(gamma), x)   # (KxN)*(NxD)-> KxD
    paran = np.sum(gamma, axis=0)  # Kx1
    paran = paran.reshape((paran.shape[0], 1))
    result = arith/paran    # KxD
    return result


# return vector with dimensions 1xK
def update_variance(gamma, x, mean):
    D = x.shape[1]
    K = mean.shape[0]
    arith = np.zeros((K, 1))
    for k in range(K):
        gamma_k = gamma[:, k]
        gamma_k = gamma_k.reshape((gamma_k.shape[0], 1))
        subtraction = np.subtract(x, mean[k])   # NxD
        # ((Nx1).*(NxD)-> NxD->sum row wise -> 1xN -> sum -> 1x1
        sub = np.sum(np.sum(np.multiply(np.power(subtraction, 2), gamma_k), axis=1))
        arith[k] = sub
    paran = D * np.sum(gamma, axis=0)   # Kx1
    paran = paran.reshape((K, 1))  # Kx1
    return arith/paran


class EM:
    def __init__(self, n_clusters, iter, tol):
        self.n_clusters = n_clusters
        self.iter = iter
        self.out = None
        self.tol = tol

    def fitting(self, x):
        mean, var, pi = init_parameters(x, self.n_clusters)
        iteration = 0

        f, m = f_logsumexp(x, mean, var, pi)
        loglikehood = update_loglikehood(f, m)

        while iteration <= self.iter:
            print('Iteration: ', iteration)
            # E-step
            gamma = update_gamma(f, m)  # NxK
            # M-step
            # update pi
            pi = (np.sum(gamma, axis=0))
            # update mean
            mean = update_mean(gamma, x)
            # update variance(var)
            var = update_variance(gamma, x, mean)
            old_loglikehood = loglikehood
            # logsumexp trick
            f, m = f_logsumexp(x, mean, var, pi)
            loglikehood = update_loglikehood(f, m)
            # check if algorithm is correct
            if loglikehood-old_loglikehood < 0:
                print('Error found in EM algorithm')
                print('Number of iterations: ', iteration)
                exit()
            # check if the convergence criterion is met
            if abs(loglikehood-old_loglikehood) < self.tol:
                print('Convergence criterion is met')
                print('Total iterations: ', iteration)
                self.mean = mean
                self.gamma = gamma
                self.var = var
                return mean, gamma, var
            # update 'safety valve' in order to not loop for an eternity
            iteration += 1
            self.mean = mean
            self.gamma = gamma
            self.var = var
        return mean, gamma, var



    def output(self, x):
        mean = self.mean.astype(np.uint8)
        max_likelihood = np.argmax(self.gamma, axis=1)
        
        means_of_data = np.array([mean[i] for i in max_likelihood])  # NxD
        means_of_data = means_of_data.astype(np.uint8)
        self.out = means_of_data

        # calculate error
        cost = cost_reconstruction(x, means_of_data)
        print('cost of reconstruction:', cost)
        
        return self.out
            

def main(filename, shape, n_clusters, tol):
    arr = load_image_asarray(filename = "./source/{}.raw".format(filename), shape = shape)

    em = EM(n_clusters = n_clusters, iter = 150, tol = tol)

    print("{}'s {} cluster with {} iteration fitting started".format(filename, n_clusters, iter))
    mean, gamma, var = em.fitting(arr)

    print("mean: {}, gamma: {}, var: {}".format(mean, gamma, var))
    output = em.output(arr).reshape(shape)

    save_dir = os.getcwd() + '/results/em/'
    os.makedirs(save_dir, exist_ok= True)
    
    img = Image.fromarray(output)
    img.save(save_dir + "{}'s_{}_cluster.bmp".format(filename, n_clusters))
    img.show()

# main("Golf", (800, 540), 2, tol = 1e-6)
# main("Golf", (800, 540), 4, tol = 1e-6)
# main("Golf", (800, 540), 8, tol = 1e-6)

# main("Gundam", (600, 600), 2, tol = 1e-6)
# main("Gundam", (600, 600), 4, tol = 1e-6)
main("Gundam", (600, 600), 8, tol = 1e-6)

# main("KU", (720, 560), 2, tol = 1e-6)
# main("KU", (720, 560), 4, tol = 1e-6)
# main("KU", (720, 560), 8, tol = 1e-6) 