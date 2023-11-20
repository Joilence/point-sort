# Point-SORT

Modified SORT tracking on 2D / 3D Point

## Install

```shell
# install as a package from GitHub
pip install git+https://github.com/Joilence/point-sort.git
```

## Usage

```python
from point_sort import track_dets

dets: np.ndarray = ...  # shape: (N, {3-5}), x, y, (z), frame, (local id)

tracker, res = track_dets(
    dets,
    dim=3,  # 2D or 3D
    min_hits=5,  # min hits to be a track
    max_age=5,  # max age to be a track
    dist_threshold: int = 50,  # distance threshold to associate dets   
)
```

More details see `example.py`.

## Development

```shell
poetry env use python3.9
poetry install

# make changes
...

poetry run make all-checks
```
