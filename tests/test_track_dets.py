import numpy as np
import pytest

from point_sort import PointSortTracker, track_dets


@pytest.fixture
def tracker_params():
    return {
        "max_age": 5,
        "min_hits": 3,
        "dist_threshold": 50,
    }


def happy_path_test_data(
    dim: int = 3, num_frame: int = 5, num_obj: int = 3, have_local_id: bool = True
) -> np.ndarray:
    """generating detections with several objects far apart and moving slowly"""
    # initialize an empty list to accumulate detections
    dets_list = []

    # create a test detections, x, y, (z), frame, local_id
    if dim == 2:
        points = np.column_stack((np.arange(num_obj), np.arange(num_obj) * 10))
    elif dim == 3:
        points = np.column_stack(
            (np.arange(num_obj), np.arange(num_obj) * 10, np.arange(num_obj) * 10)
        )
    else:
        raise ValueError(f"Unsupported dimension {dim}, only 2 or 3 is supported")

    for frame in range(num_frame):
        # Add points with frame and local_id if required
        for local_id, p in enumerate(points):
            det = (
                np.hstack((p, frame, local_id + 1))
                if have_local_id
                else np.hstack((p, frame))
            )
            dets_list.append(det)

        # move every point by 1 unit in the x-direction
        points[:, 1] += 1

    return np.vstack(dets_list)


def every_track_id_has_unique_local_id(track_res: np.ndarray) -> bool:
    """check if each track_id only corresponds to one unique local_id"""
    for track_id in np.unique(track_res[:, 1]):
        local_ids = track_res[track_res[:, 1] == track_id, -1]
        if len(np.unique(local_ids)) != 1:
            return False
    return True


# Happy path tests with various realistic test values
@pytest.mark.parametrize(
    "dets, dim, test_id",
    [
        (happy_path_test_data(3, 5, 5, True), 3, "3d_with_local_id"),
        (happy_path_test_data(3, 5, 5, False), 3, "3d_without_local_id"),
        (happy_path_test_data(2, 5, 5, True), 2, "2d_with_local_id"),
        (happy_path_test_data(2, 5, 5, False), 2, "2d_without_local_id"),
    ],
    ids=str,
)
def test_track_dets_happy_path(dets, dim, test_id, tracker_params):
    # Act
    tracker, res = track_dets(dets, dim=dim, **tracker_params)

    # Assert
    # return type
    assert isinstance(
        tracker, PointSortTracker
    ), f"Test {test_id}: Tracker is not an instance of PointSortTracker"
    assert isinstance(
        res, np.ndarray
    ), f"Test {test_id}: Result is not an instance of np.ndarray"
    # the column number of the result should be input columns + 1 (extra id column)
    assert (
        res.shape[1] == dets.shape[1] + 1
    ), f"Test {test_id}: Result shape[1] {res.shape[1]} does not match input shape[1] {dets.shape[1]}+1"
    # the rows (dets) should be less than or equal to the input
    assert (
        res.shape[0] <= dets.shape[0]
    ), f"Test {test_id}: Result shape[0] {res.shape[0]} is greater than input shape[0] {dets.shape[0]}"

    assert dets.shape[1] == dim + 1 or every_track_id_has_unique_local_id(
        dets
    ), f"Test {test_id}: tracking result does not have unique local_id for each track_id"


# Edge cases
@pytest.mark.parametrize(
    "dets, dim, test_id",
    [
        (np.empty((0, 5)), 3, "empty_3d_with_local_id"),
        (np.empty((0, 4)), 2, "empty_2d_with_local_id"),
        (np.empty((0, 4)), 3, "empty_3d_without_local_id"),
        (np.empty((0, 3)), 2, "empty_2d_without_local_id"),
    ],
    ids=str,
)
def test_track_dets_empty_input(dets, dim, test_id, tracker_params):
    # Act
    tracker, res = track_dets(dets, dim=dim, **tracker_params)

    # Assert
    assert res is None, f"Test {test_id}: Result is not None"
    assert tracker is None, f"Test {test_id}: Tracker is not None"
