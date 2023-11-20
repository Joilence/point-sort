#!/usr/bin/env python3
import numpy as np

from point_sort import track_dets

num_objects = 3
num_frames = 5
print(f"num_objects: {num_objects}, num_frames: {num_frames}")

# create a test dets [x, y, z, frame]
dets_list = []

points = [[n * 10, n * 10, n * 10] for n in range(num_objects)]

for frame in range(num_frames):
    # add points
    for p in points:
        det = np.array([*p, frame])
        dets_list.append(det)

    # move every point by 1 unit in the x-direction
    points = [[p[0], p[1] + 1, p[2]] for p in points]

dets = np.vstack(dets_list)
print(f"dets: (x, y, z, frame)\n{dets}")

# track the points
tracks, res = track_dets(dets, dim=3)
print(f"res: (frame, track_id, x, y, z)\n{res}")
