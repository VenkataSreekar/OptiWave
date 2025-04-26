from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from math import log10
import math
import random
import Block_SVD as svd

def svd_bpso(image, p_n = None, n_iterations = None)
    # Check whether the input matrix is colour or gray
    if len(image.shape)>2:
        X = image.dot([0.299, 0.5870, 0.114])
    elif len(image.shape) = 2:
        X = image
    # Initializing number of particles and iterations if they aren't specified
    if p_n is None:
        p_n = 30
    if n_iteration is None:
        n_iterations = 5
        
    # Apply SVD on the input matrix X
    U, sig, VT = svd.Block_SVD(X)
    sig = np.diag(sig)

    #initializing PSO parameters
    c1 = 2 # Cognitive component
    c2 = 2 # Social component

    # Algorithm created to make the Search-Space compact resulting in increased computational efficiency
    k_end = int(max(X.shape[0],X.shape[1])/(int(min(X.shape[0],X.shape[1])**0.5)))

    # Defining a function to calculate PSNR
    def compressed_img(Xapr,k):
        U_k = U[:, :k]
        sig_k = sig[:k, :k]
        VT_k = VT[:k, :]
        cmpd_img = np.dot(U_k, np.dot(sig_k, VT_k))
    # Using Mean Squared Error to define PSNR
        MSE = ((X-cmpd_img)**2).mean(axis=None)
        PSNR = 10*log10(((255)**2)/MSE)
        if PSNR>0:
            return PSNR
        return -PSNR

    # Initializing a random gbest to make the program run
    O = list(range(k_end,k_end+100))
    gbest = random.choice(O)
    k= gbest

    # Defining a Fitness function
    def fitness(X,k):
        k = k
        Xapr = X
        PSNR=compressed_img(Xapr,k)
        compression_ratio = k*(1+X.shape[0] + X.shape[1])/(X.shape[0]*X.shape[1])
        fitness = ((PSNR) +2/compression_ratio)*10
        return fitness
    # Initializing particles, their positions and velocities inside of the boundary    
    class particles:
        def __init__(self,bounds):
            bounds = (k_end,k_end+100)
            self.position =random.randint(bounds[0],bounds[1])
            self.velocity = 0
            self.best_position = self.position

    # Setting up the Search-Space for particles
    bounds = (k_end,k_end+100)
    swarm = [particles(bounds) for _ in range(p_n)]

    # Main PSO loop
    for t in range(n_iterations):
        for i in swarm:
            sum_pbest=0
            for j in swarm:
                sum_pbest = sum_pbest+j.best_position
            current_fitness = fitness(X, i.position)
            pbest_fitness = fitness(X, i.best_position)
            
            # Calculation of personal best position
            if current_fitness < pbest_fitness:
                i.best_position = i.position
            
            # Calculation of best position of swarm
            if current_fitness < fitness(X,gbest):
                gbest = i.position

            # Setting random values of Cognitive and Social components to avoid quick convergence
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)

            # Probability with which the particle would head towards the best position of swarm
            P_t = gbest/sum_pbest

            # Velocity update equation
            i.velocity = (1-t/n_iterations)*i.velocity + (1-P_t)*(math.exp(-1+t/n_iterations))*r1*c1*(i.best_position-i.position) + P_t*r2*c2*(gbest-i.position)

            # Position update equation
            i.position = i.position + int(math.floor(i.velocity+0.5))

            # Fixing particles inside of boundary
            if i.position < bounds[0]:
                i.position = bounds[0]
            elif i.position > bounds[1]:
                i.position = bounds[1]

    # Reconstruction image matrix and returning it
    Xcmp = np.dot(U[:,:gbest], np.dot(sig[:gbest, :gbest], VT[:gbest, :]))
    return Xcmp