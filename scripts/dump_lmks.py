import os
import sys
import cv2
import toml
import pickle
import numpy as np
from tqdm import tqdm

toml_path = sys.argv[1]
with open(toml_path) as fp:
    data = toml.load(fp)
video_dir = os.path.dirname(toml_path)


def _iframe_to_ts(i):
    ts = data['frames'][i]['ms']
    if ts <= 0:
        ts = i * 1000.0 / data['fps']
    return ts


iframe = 0
mapping = dict()
for i in tqdm(range(int(sys.argv[2]))):
    ts = i * 1000.0 / 25.0 - 60
    while iframe < len(data['frames']) and _iframe_to_ts(iframe) <= ts:
        iframe += 1
    jframe = np.clip(iframe, 0, len(data['frames']) - 1)
    iframe = np.clip(iframe - 1, 0, len(data['frames']) - 1)
    pts0 = np.asarray(data['frames'][iframe]['points'], dtype=np.float32)
    pts1 = np.asarray(data['frames'][jframe]['points'], dtype=np.float32)
    ts0 = _iframe_to_ts(iframe)
    ts1 = _iframe_to_ts(jframe)
    a = (ts - ts0) / (ts1 - ts0)
    a = np.clip(a, 0, 1)
    assert 0 <= a <= 1
    pts = pts0 * (1-a) + pts1 * a
    # pts = pts0 if a < 0.5 else pts1
    # resize image
    img = cv2.imread(os.path.join(video_dir, "full", f"{i+1:05d}.png"))
    pts[:, 0] /= img.shape[1]
    pts[:, 1] /= img.shape[0]
    img = cv2.resize(img, (256, 256))
    pts *= 256
    path = os.path.join(video_dir, "crop", f"{i+1:05d}.png")
    cv2.imwrite(path, img)
    mapping[path] = pts

    for p in pts:
        c = (int(p[0]), int(p[1]))
        cv2.circle(img, c, 2, (0, 255, 0), -1)
    cv2.imshow('canvas', cv2.resize(img, (512, 512)))
    cv2.waitKey(1)

with open(os.path.join(video_dir, "landmark.pkl"), "wb") as fp:
    pickle.dump(mapping, fp)
