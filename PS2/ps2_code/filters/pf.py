"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)
        
        # TODO add here specific class variables for the PF
        self.num_particles = num_particles
        self.X = np.random.multivariate_normal(self.mu, self.Sigma, self.num_particles)
        self.weights = np.zeros(self.num_particles)

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for i in range(self.num_particles):
            self.X[i, :] = sample_from_odometry(self.X[i, :], u, self._alphas)
            self.X[i, 2] = wrap_angle(self.X[i, 2])
        
        self._state_bar = get_gaussian_statistics(self.X)

    def update(self, z):
        # TODO implement correction step
        ID = int(z[1])
        z_obs = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            z_obs[i] = get_observation(self.X[i], ID)[0]
            self.weights[i] = gaussian.pdf(z_obs[i], loc = z[0], scale = self._Q ** 0.5)
        
        self.weights += 10 ** (-200)
        self.weights /= np.sum(self.weights)

        R_temp = uniform(low = 0, high = 1 / self.num_particles)
        temp = self.weights[0]
        ind = 0
        X = np.zeros([self.num_particles, self.state_dim])

        for i in range(self.num_particles):
            U = R_temp + i / self.num_particles

            while U > temp:
                ind += 1
                temp += self.weights[ind]

            X[i, :] = self.X[ind, :]

        self.X = X
        self._state = get_gaussian_statistics(self.X)
