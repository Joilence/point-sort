from typing import Tuple

import numpy as np


def linear_assignment(dist_matrix: np.ndarray) -> np.ndarray:
    """Solve the linear assignment problem using the Hungarian algorithm.
    :param: dist_matrix: np.array of shape (n, m), [[dist_1, dist_2, ..., dist_m], ...]
    :return: np.array of shape (n, 2), [[row_1, col_1], [row_2, col_2], ...]
    """
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return np.array(list(zip(row_ind, col_ind)))


def compute_distance(dets: np.ndarray, preds: np.ndarray, dim: int) -> np.ndarray:
    """Compute distance between each pair of the two collections of inputs.
    :param: dets: np.array of shape (n, 5), [[x, y, z, frame, in_frame_id (local id)], ...]
    :param: preds: np.array of shape (m, 3), [[x, y, z], ...]
    :param: dim: int, dimension of the point
    :return: np.array of shape (n, m), [[dist_1, dist_2, ..., dist_m], ...]
    """
    return np.sqrt(
        np.sum((dets[:, np.newaxis, :dim] - preds[np.newaxis, :, :dim]) ** 2, axis=2)
    )


def associate_detections_to_predictions(
    dets: np.ndarray,
    preds: np.ndarray,
    dim: int,
    dist_threshold: float = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assigns detections to tracked object (both represented as points)
    :param: dets: np.array of shape (n, n_dim + 2), [[x, y, z ... {n_dim}, frame, in_frame_id (local id)], ...]
    :param: preds: np.array of shape (m, n_dim + 1), [[x, y, z ... {n_dim}, score], ...]
    :param: dim: int, dimension of the points
    :param: dist_threshold: float, distance threshold
    :return: matches: np.array of shape (n, 2), [[det_idx, pred_idx], ...]
    :return: unmatched_dets: np.array of shape (n, ), [det_idx, ...]
    :return: unmatched_preds: np.array of shape (m, ), [pred_idx, ...]
    """
    if len(preds) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(dets)),
            np.empty((0, dim + 1), dtype=int),
        )

    dist_matrix = compute_distance(dets, preds, dim)

    if min(dist_matrix.shape) > 0:
        matched_indices = linear_assignment(dist_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_dets = [d for d, det in enumerate(dets) if d not in matched_indices[:, 0]]
    unmatched_preds = [
        t for t, trk in enumerate(preds) if t not in matched_indices[:, 1]
    ]
    # filter out matched with high distance
    matches = []
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(m[0])
            unmatched_preds.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    return (
        (np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)),
        np.array(unmatched_dets),
        np.array(unmatched_preds),
    )
