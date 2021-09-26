import os
import sys
import cv2
import toml
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import rmtree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_vocaset_videos(output_root, data_root, speaker, training, dest_size=256, debug=False):
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, speaker, "train" if training else "test")
    data_root = os.path.expanduser(data_root)
    data_root = os.path.join(data_root, speaker)

    seq_range = list(range(0, 20)) if training else list(range(20, 40))
    for i_seq in tqdm(seq_range, desc=f"{speaker}/{seq_range[0]}-{seq_range[-1]}"):
        out_dir = os.path.join(output_root, f"clip{i_seq:02d}")
        # skip if already done before
        if os.path.exists(os.path.join(out_dir, "landmark.pkl")):
            continue

        # prepare output dirs
        os.makedirs(os.path.join(out_dir, "full"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "crop"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "feature"), exist_ok=True)

        # data source
        prefix = os.path.join(data_root, f"sentence{i_seq+1:02d}")
        vpath = prefix + ".mp4"
        lpath = prefix + "-lmks-ibug-68.toml"

        # 1. -> 25 fps images and audio
        assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} -r 25 {out_dir}/full/%05d.png") == 0
        assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} {out_dir}/audio/audio.wav") == 0

        # 2. audio fetures
        assert os.system(f"{sys.executable} {ROOT}/vendor/ATVGnet/code/test.py -i {out_dir}/") == 0

        # 3. resize and dump landmarks
        with open(lpath) as fp:
            lmks_data = toml.load(fp)
        lmks_mapping = dict()
        i_lmk = 0

        def _i_lmk_to_ts(i):
            ts = lmks_data['frames'][i]['ms']
            if ts <= 0:
                ts = i * 1000.0 / lmks_data['fps']
            return ts

        img_list = sorted(glob(f"{out_dir}/full/*.png"))
        for i_frm, img_path in enumerate(img_list):
            save_path = f"{out_dir}/crop/{os.path.basename(img_path)}"
            img = cv2.imread(img_path)
            img = cv2.resize(img, (dest_size, dest_size))
            cv2.imwrite(save_path, img)
            # fetch lmk
            ts = i_frm * 1000.0 / 25.0 - 60  # HACK
            while i_lmk < len(lmks_data['frames']) and _i_lmk_to_ts(i_lmk) <= ts:
                i_lmk += 1
            jframe = np.clip(i_lmk, 0, len(lmks_data['frames']) - 1)
            iframe = np.clip(i_lmk - 1, 0, len(lmks_data['frames']) - 1)
            pts0 = np.asarray(lmks_data['frames'][iframe]['points'], dtype=np.float32)
            pts1 = np.asarray(lmks_data['frames'][jframe]['points'], dtype=np.float32)
            ts0 = _i_lmk_to_ts(iframe)
            ts1 = _i_lmk_to_ts(jframe)
            if np.isclose(ts0, ts1):
                a = 1
            else:
                a = (ts - ts0) / (ts1 - ts0)
            a = np.clip(a, 0, 1)
            assert 0 <= a <= 1
            pts = pts0 * (1-a) + pts1 * a
            pts[:, 0] = pts[:, 0] / lmks_data['resolution'][0] * dest_size
            pts[:, 1] = pts[:, 1] / lmks_data['resolution'][1] * dest_size
            # update mapping
            lmks_mapping[save_path] = pts
            # debug
            if debug:
                for p in pts:
                    c = (int(p[0]), int(p[1]))
                    cv2.circle(img, c, 2, (0, 255, 0), -1)
                cv2.imshow('img', img)
                cv2.waitKey(1)
        # remove full
        rmtree(f"{out_dir}/full")
        # save landmarks
        with open(os.path.join(out_dir, "landmark.pkl"), "wb") as fp:
            pickle.dump(lmks_mapping, fp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=f"{ROOT}/data/vocaset_video")
    parser.add_argument("--source_dir", type=str, default="~/assets/vocaset/Data/videos_lmks")
    parser.add_argument("--dest_size", type=int, default=256)
    parser.add_argument("--speakers", type=str, nargs="+", default=["FaceTalk_170725_00137_TA"])
    args = parser.parse_args()

    for spk in args.speakers:
        prepare_vocaset_videos(args.output_dir, args.source_dir, spk, dest_size=args.dest_size, training=True)
        prepare_vocaset_videos(args.output_dir, args.source_dir, spk, dest_size=args.dest_size, training=False)
