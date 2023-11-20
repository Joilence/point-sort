from typing import List

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanPointNDTracklet(object):
    """This class represents the internal state of individual tracked objects observed as 3d point."""

    count = 0

    def __init__(self, point: np.ndarray, dim: int):
        """Initialises a tracklet using initial point.
        :param: point: np.array of shape (dim + 2,), [x, y, z ... {n_dim} , frame, in_frame_id (local id)]
        """
        self.dim = dim  # dimension of the point
        self.init_kalman_filter(point)  # init kf
        self.time_since_update = 0  # number of frames since last measurement update
        self.id = (
            KalmanPointNDTracklet.count
        )  # id of the tracklet assigned by class counter
        KalmanPointNDTracklet.count += 1  # increase count by 1
        self.history: List[np.ndarray] = []  # history of the tracklet
        self.hits = 0  # number of total hits including the first detection
        self.hit_streak = 0  # number of continuing hit considering the first detection
        self.age = 0  # number of frames since first detection
        self.local_ids_frame = [point[-2]]  # frame id of the local id
        self.local_ids = [point[-1]]  # local id

    def init_kalman_filter(self, point: np.ndarray):
        """Initialises a Kalman Filter using initial point.
        :param: point: np.array of shape (dim + 2,), [x, y, z, ... {n_dim}, frame, in_frame_id (local id)]
        """

        pts_dim = self.dim
        dim_x = pts_dim * 2
        dim_z = pts_dim

        # define constant velocity model
        # dim_x is the number of state variables for the Kalman filter.
        # position and velocity (x, y, ... {n_dim}, vx, vy, ... v{n_dim})
        # dim_z is the number of measurement inputs. position: (x, y, ... {n_dim})
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # self.kf.x: state transition matrix, in class default init as eye(dim_x)
        # TODO: understand why init kf.F like this
        self.kf.F = np.eye(dim_x)
        for i in range(dim_x - dim_z):
            self.kf.F[i, i + dim_z] = 1
        # self.kf.F = np.array(
        #     [[1, 0, 0, 1, 0, 0],
        #      [0, 1, 0, 0, 1, 0],
        #      [0, 0, 1, 0, 0, 1],
        #      [0, 0, 0, 1, 0, 0],
        #      [0, 0, 0, 0, 1, 0],
        #      [0, 0, 0, 0, 0, 1]]
        # )

        # self.kf.H: measurement function, in class default init as zeros((dim_z, dim_x))
        # TODO: understand why init kf.H like this
        self.kf.H = np.zeros((dim_z, dim_x))
        for i in range(dim_z):
            self.kf.H[i, i] = 1
        # self.kf.H = np.array(
        #     [[1, 0, 0, 0, 0, 0],
        #      [0, 1, 0, 0, 0, 0],
        #      [0, 0, 1, 0, 0, 0]]
        # )

        # self.kf.R: state uncertainty, in class default init as eye(dim_z)
        # self.kf.R[2:, 2:] *= 10.

        # self.kf.P: uncertainty covariance matrix, in class default init as eye(dim_x)
        self.kf.P[
            pts_dim:, pts_dim:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        # self.kf.Q: process uncertainty, in class default init as eye(dim_x)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[pts_dim:, pts_dim:] *= 0.01

        # self.kf.x: state vector, in class default init as zeros((dim_x, 1))
        self.kf.x[:pts_dim] = point[:pts_dim].reshape(pts_dim, 1)

    def update(self, point: np.ndarray):
        """Updates the state vector with observed point.
        :param: point: np.array of shape (5,), [x, y, z ... {n_dim}, frame, in_frame_id (local id)]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(point[: self.dim].reshape(self.dim, 1))
        self.local_ids.append(point[-1])
        self.local_ids_frame.append(point[-2])

    def predict(self):
        """Advances the state vector and returns the predicted point."""
        if (self.kf.x[self.dim :] == 0).all():
            self.kf.x[self.dim :] *= 0.1
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[: self.dim])
        return self.history[-1]

    def get_state(self):
        """Returns the current point estimate."""
        return self.kf.x[: self.dim].reshape(-1, self.dim)
