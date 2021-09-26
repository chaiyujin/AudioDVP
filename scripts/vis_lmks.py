import os
import sys
import cv2
import pickle


# find clips
clip_dirs = []
for dirpath, subdirs, _ in os.walk("data/vocaset_video"):
    for subdir in subdirs:
        if subdir.startswith("clip") and os.path.exists(os.path.join(dirpath, subdir, "crop")):
            clip_dirs.append(os.path.join(dirpath, subdir))
clip_dirs = sorted(clip_dirs)

for data_dir in clip_dirs:
    with open(os.path.join(data_dir, "landmark.pkl"), "rb") as fp:
        lmks_dict = pickle.load(fp)

    for k in lmks_dict:
        img = cv2.imread(k)
        pts = lmks_dict[k]
        for p in pts:
            c = (int(p[0]), int(p[1]))
            cv2.circle(img, c, 2, (0, 255, 0), -1)
        cv2.imshow('landmarks - 68', cv2.resize(img, (512, 512)))
        cv2.waitKey(40)
