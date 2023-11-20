import numpy as np

from point_sort.KalmanPointNDTracklet import KalmanPointNDTracklet
from point_sort.utility import associate_detections_to_predictions


class PointSortTracker:
    """Tracker based on SORT algorithm:
    - apply on 3D points instead of bounding boxes
    - use distance Euclidean distance to match 3D points instead of IoU
    """

    def __init__(
        self,
        dim: int,
        max_age: int = 1,
        min_hits: int = 3,
        dist_threshold=30,
    ):
        """Sets key parameters for SORT
        :param max_age: int, maximum number of frames to keep alive a track without associated detections.
        :param min_hits: int, minimum number of associated detections before track is initialised.
        :param dist_threshold: int, distance threshold to match 3D points
        :param ref_cam_name: str, reference camera name, used to record local ids
        """
        self.dim = dim
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold

        self.live_tracklets: list[KalmanPointNDTracklet] = []  # active tracklets
        self.dead_tracklets: list[KalmanPointNDTracklet] = []  # inactive tracklets

        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        :param dets: np.array of shape (n, dim + 2), [[x, y, z ... {n_dim}, frame, in_frame_id (local id), ], ...]
        :return: np.array of shape (n, dim + 2), [[x, y, z ... {n_dim}, score, id], ...]
        requires: this method must be called once for each frame even with empty detections.
        note: the number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get tracklet predictions
        preds = np.zeros(
            (len(self.live_tracklets), self.dim + 1)
        )  # [x, y, z ... {n_dim}, score]
        trkr_ind_to_del = []
        for t, pred in enumerate(preds):
            pos = self.live_tracklets[t].predict().reshape(-1, self.dim)[0]
            # pred[:] = [pos[0], pos[1], pos[2], 0] # TODO: add score
            pred[: self.dim] = pos[: self.dim]
            pred[-1] = 0
            # replace last line 0, 1, 2 ... with pos[:self.dim]

            if np.any(np.isnan(pos)):
                trkr_ind_to_del.append(t)
        # remove invalid rows with values of nan or inf
        preds = np.ma.compress_rows(np.ma.masked_invalid(preds))
        # remove invalid tracklets by index from the tail
        for t in reversed(trkr_ind_to_del):
            self.dead_tracklets.append(self.live_tracklets.pop(t))
        matched, unmatched_dets, unmatched_preds = associate_detections_to_predictions(
            dets, preds, self.dim, self.dist_threshold
        )

        # update matched tracklet predictions with assigned detections
        for m in matched:
            trkr_ind = m[1]  # tracklet index
            matched_det = dets[m[0], :]  # matched detection state
            self.live_tracklets[trkr_ind].update(matched_det)

        # create and initialise new tracklets for unmatched detections
        for i in unmatched_dets:
            klm_tkl = KalmanPointNDTracklet(
                dim=self.dim, point=dets[i, :]
            )  # create a new tracklet
            self.live_tracklets.append(klm_tkl)  # add to list of tracklets

        ret = []
        i = len(self.live_tracklets)
        for klm_tkl in reversed(self.live_tracklets):
            d = klm_tkl.get_state()[0]
            if (klm_tkl.time_since_update < 1) and (
                klm_tkl.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    # track_id = id + 1 for compatibility with MOTChallenge
                    np.concatenate(
                        # state (coord), track_id, local_id
                        (d, [klm_tkl.id + 1, klm_tkl.local_ids[-1]])
                    ).reshape(1, -1)
                )
            i -= 1
            # remove dead tracklet
            if klm_tkl.time_since_update > self.max_age:
                self.dead_tracklets.append(self.live_tracklets.pop(i))

        return np.concatenate(ret) if ret else np.empty((0, self.dim + 2))

    def collect_tracklets(self) -> list[KalmanPointNDTracklet]:
        """Collect qualified tracklets with hits less than min_hits
        :return: list of KalmanPointTracklet
        """
        return [
            tracklet
            for tracklet in self.live_tracklets + self.dead_tracklets
            if tracklet.hits >= self.min_hits
        ]
