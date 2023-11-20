""" This module implements a multi-object tracker for 2D / 3D points, inspired by SORT. """

import logging
from typing import Optional, Tuple

import numpy as np
from tqdm import trange

from point_sort.PointSortTracker import PointSortTracker

logger = logging.getLogger(__name__)


def track_dets(
    dets: np.ndarray,
    dim: int = 3,
    max_age: int = 5,
    min_hits: int = 5,
    dist_threshold: int = 50,
) -> Tuple[Optional[PointSortTracker], Optional[np.ndarray]]:
    """Track a sequence of point detections in 2D or 3D.
    Optional column of local id, used to verify tracking results on ground truth.
    If no local id is provided, a dummy local id of -1 will be added, and removed from the output.
    :param tracker: Tracker object
    :param dets: np.ndarray, detections, accepting two formats:
        1. (n, dim + 2) array, where n is the number of detections, columns are x, y, (z), frame, local_id
        2. (n, dim + 1) array, where n is the number of detections, columns are x, y, (z), frame
    :return: tracker, np.ndarray, tracking results, (n, dim + 3) array, columns are frame, track_id, x, y, (z), local_id
    """

    # deal with empty input
    if len(dets) == 0:
        logger.warning("Empty input dets.")
        return None, None

    # verify input
    assert dim in {2, 3}, f"Only support dim=2 or dim=3, got {dim}."
    assert dets.shape[1] in [
        dim + 2,  # x, y, (z), frame, local_id
        dim + 1,  # x, y, (z), frame
    ], f"Input dets for dim={dim} should be either (n, {dim + 2}) or (n, {dim + 1})."

    # add dummy local id if not provided
    no_local_id = dets.shape[1] == dim + 1
    if no_local_id:  # add dummy local id -1
        dets = np.concatenate((dets, np.ones((dets.shape[0], 1)) * -1), axis=1)

    # create tracker
    tracker = PointSortTracker(
        dim=dim,
        max_age=max_age,
        min_hits=min_hits,
        dist_threshold=dist_threshold,
    )

    # track
    res = np.empty((0, dim + 3))  # frame, track_id, x, y, (z), local_id
    min_frame = int(dets[:, -2].min())
    max_frame = int(dets[:, -2].max())
    for frame in trange(
        min_frame,
        max_frame + 1,
        desc="Tracking",
        bar_format="{desc}: {percentage:3.0f}%|{bar:10}{r_bar}",
    ):
        dets_input = dets[dets[:, -2] == frame]
        if len(dets_input) == 0:
            logger.warning(f"No detections in frame {frame}.")
        klm_states = tracker.update(dets_input)

        for state in klm_states:
            coord = state[:dim]
            track_id = state[-2]
            local_id = state[-1]
            res = np.append(
                res,
                # frame, track_id, x, y, (z), local_id
                np.concatenate(([frame, track_id], coord, [local_id])).reshape(1, -1),
                axis=0,
            )

    if no_local_id:
        res = res[:, :-1]  # remove dummy local id

    return tracker, res
