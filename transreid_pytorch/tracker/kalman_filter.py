import numpy as np
import scipy.linalg
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space is
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        
        # We can optionally instantiate a FilterPy object here or use it per step,
        # but since DeepSORT's implementation is basically just dot products,
        # we can just write the predict/update steps or use FilterPy.
        # To strictly use filterpy as requested:
        kf = FilterPyKalmanFilter(dim_x=8, dim_z=4)
        kf.F = self._motion_mat
        kf.H = self._update_mat
        kf.x = mean.reshape(-1, 1)
        kf.P = covariance
        return kf

    def predict(self, kf):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        kf : FilterPyKalmanFilter
            The kalman filter object for a track.
        """
        std_pos = [
            self._std_weight_position * kf.x[3, 0],
            self._std_weight_position * kf.x[3, 0],
            1e-2,
            self._std_weight_position * kf.x[3, 0]]
        std_vel = [
            self._std_weight_velocity * kf.x[3, 0],
            self._std_weight_velocity * kf.x[3, 0],
            1e-5,
            self._std_weight_velocity * kf.x[3, 0]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        kf.Q = motion_cov
        kf.predict()

    def update(self, kf, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        kf : FilterPyKalmanFilter
            The kalman filter object for a track.
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h).
        """
        std = [
            self._std_weight_position * kf.x[3, 0],
            self._std_weight_position * kf.x[3, 0],
            1e-1,
            self._std_weight_position * kf.x[3, 0]]
        innovation_cov = np.diag(np.square(std))
        
        kf.R = innovation_cov
        kf.update(measurement.reshape(-1, 1))

    def project(self, kf):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * kf.x[3, 0],
            self._std_weight_position * kf.x[3, 0],
            1e-1,
            self._std_weight_position * kf.x[3, 0]]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, kf.x).flatten()
        covariance = np.linalg.multi_dot((
            self._update_mat, kf.P, self._update_mat.T))
        return mean, covariance + innovation_cov
