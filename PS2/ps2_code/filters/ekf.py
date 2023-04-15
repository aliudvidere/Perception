"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def G_t(self, mu, u):
        return np.array([[1, 0, (-u[1] * np.sin(u[0] + mu[2]))[0]],
                         [0, 1,  (u[1] * np.cos(u[0] + mu[2]))[0]],
                         [0, 0,                                 1]])

    def V_t(self, mu, u):
        return np.array([[(-u[1] * np.sin(u[0] + mu[2]))[0], (np.cos(u[0] + mu[2]))[0], 0],
                         [( u[1] * np.cos(u[0] + mu[2]))[0], (np.sin(u[0] + mu[2]))[0], 0],
                         [1,                                 0,                         1]])

    def H_t(self, mu, ID):
        mx = self._field_map.landmarks_poses_x[ID]
        my = self._field_map.landmarks_poses_y[ID]
        mux = mu[0, 0]
        muy = mu[1, 0]
        return np.array([(my - muy) / ((mx-mux)**2 + (my-muy)**2), -(mx - mux) / ((mx-mux)**2 + (my-muy)**2), -1])

    def R_t(self, V_t, M_t):
        return V_t @ M_t @ V_t.T

    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

        V_t = self.V_t(self._state_bar.mu, u)
        G_t = self.G_t(self._state_bar.mu, u)
        M_t = get_motion_noise_covariance(u, self._alphas)
        R_t = self.R_t(V_t, M_t)
        

        self._state_bar.mu = get_prediction(self._state_bar.mu[:, 0], u)[np.newaxis].T
        self._state_bar.mu[2, 0] = wrap_angle(self._state_bar.mu[2, 0])

        self._state_bar.Sigma = G_t @ self._state_bar.Sigma @ G_t.T + R_t

        



    def update(self, z):
        # TODO implement correction step
        ID = int(z[1])
        z_obs = get_expected_observation(self.mu_bar, ID)

        H_t = self.H_t(self._state_bar.mu, ID)

        S_t = H_t @ self._state_bar.Sigma @ H_t.T + self._Q * 2

        K_t = self._state_bar.Sigma @ H_t.T * S_t ** (-1)

        self._state.mu = self._state_bar.mu + (K_t * wrap_angle(z[0] - z_obs[0]))[np.newaxis].T
        self._state.mu[2, 0] = wrap_angle(self._state.mu[2, 0])

        self._state.Sigma = np.asarray((np.eye(3) - np.asmatrix(K_t).T @ np.asmatrix(H_t)) @ self._state_bar.Sigma)


