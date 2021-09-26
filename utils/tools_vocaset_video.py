import os
import sys
sys.path.append(".")

import cv2
import toml
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from models import networks
from utils import util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_vocaset_video(output_root, data_root, speaker, training, dest_size=256, debug=False):
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, speaker, "train" if training else "test")
    data_root = os.path.expanduser(data_root)
    data_root = os.path.join(data_root, speaker)

    seq_range = list(range(0, 20)) if training else list(range(20, 40))
    for i_seq in tqdm(seq_range, desc=f"[prepare_vocaset_video]: {speaker}/{seq_range[0]}-{seq_range[-1]}"):
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


def visualize_reconstruction(dataset_dir, speaker):
    # find clips
    clip_dirs = []
    for dirpath, subdirs, _ in os.walk(os.path.join(dataset_dir, speaker)):
        for subdir in subdirs:
            if subdir.startswith("clip") and os.path.exists(os.path.join(dirpath, subdir, "crop")):
                clip_dirs.append(os.path.join(dirpath, subdir))
    clip_dirs = sorted(clip_dirs)

    for clip_dir in tqdm(clip_dirs, desc="Visualize reconstruction"):
        if os.path.exists(clip_dir + "-debug.mp4"):
            continue
        cmd = (
            "ffmpeg -hide_banner -n -loglevel error "
            "-thread_queue_size 8192 -i {0}/render/%05d.png "
            "-thread_queue_size 8192 -i {0}/crop/%05d.png "
            "-thread_queue_size 8192 -i {0}/overlay/%05d.png "
            "-i {0}/audio/audio.wav "
            "-filter_complex hstack=inputs=3 -vcodec libx264 -preset slower "
            "-profile:v high -crf 18 -pix_fmt yuv420p {0}-debug.mp4"
        ).format(clip_dir)
        assert os.system(cmd) == 0


def build_nfr_dataset(mouth_mask, dataset_dir, speaker):
    data_dir = os.path.join(dataset_dir, speaker)
    done_flag = os.path.join(data_dir, "nfr", "done.flag")
    if os.path.exists(done_flag):
        print("[build_nfr_dataset]: Find {}. It's already done! Skip.".format(done_flag))
        return

    # find training clips
    clip_dirs = util.find_clip_dirs(data_dir, with_train=True, with_test=False)

    # generate mask for all clips
    for clip_dir in tqdm(clip_dirs, desc="[build_nfr_dataset]: Generate masks"):
        util.create_dir(os.path.join(clip_dir, 'mask'))

        alpha_list = util.load_coef(os.path.join(clip_dir, 'alpha'      ), verbose=False)
        beta_list  = util.load_coef(os.path.join(clip_dir, 'beta'       ), verbose=False)
        delta_list = util.load_coef(os.path.join(clip_dir, 'delta'      ), verbose=False)
        gamma_list = util.load_coef(os.path.join(clip_dir, 'gamma'      ), verbose=False)
        angle_list = util.load_coef(os.path.join(clip_dir, 'rotation'   ), verbose=False)
        trnsl_list = util.load_coef(os.path.join(clip_dir, 'translation'), verbose=False)

        for i in tqdm(range(len(alpha_list)), leave=False):
            alpha = alpha_list[i].unsqueeze(0).cuda()
            beta  = beta_list [i].unsqueeze(0).cuda()
            delta = delta_list[i].unsqueeze(0).cuda()
            gamma = gamma_list[i].unsqueeze(0).cuda()
            rotat = angle_list[i].unsqueeze(0).cuda()
            trnsl = trnsl_list[i].unsqueeze(0).cuda()

            mask = mouth_mask(alpha, delta, beta, gamma, rotat, trnsl)
            mask = mask.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255.0
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4)

            cv2.imwrite(os.path.join(clip_dir, 'mask', '%05d.png' % (i+1)), mask)

    # create dir for nfr dataset
    util.create_dir(os.path.join(data_dir, 'nfr', 'A', 'train'))
    util.create_dir(os.path.join(data_dir, 'nfr', 'B', 'train'))

    # collect all masks, crops and renders from clips
    masks, crops, renders = [], [], []
    for clip_dir in clip_dirs:
        masks  .extend(util.get_file_list(os.path.join(clip_dir, 'mask')))
        crops  .extend(util.get_file_list(os.path.join(clip_dir, 'crop')))
        renders.extend(util.get_file_list(os.path.join(clip_dir, 'render')))

    # save into nfr dataset
    for i in tqdm(range(len(masks)), desc="[build_nfr_dataset]: Write into A/train or B/train"):
        mask   = cv2.imread(masks[i])
        crop   = cv2.imread(crops[i])
        render = cv2.imread(renders[i])

        masked_crop   = cv2.bitwise_and(crop, mask)
        masked_render = cv2.bitwise_and(render, mask)

        cv2.imwrite(os.path.join(data_dir, 'nfr', 'A', 'train', '%05d.png' % (i+1)), masked_crop)
        cv2.imwrite(os.path.join(data_dir, 'nfr', 'B', 'train', '%05d.png' % (i+1)), masked_render)

    # create AB dataset from A, B dirs
    for sp in os.listdir(os.path.join(data_dir, 'nfr', 'A')):
        image_fold_A = os.path.join(os.path.join(data_dir, 'nfr', 'A'), sp)
        image_fold_B = os.path.join(os.path.join(data_dir, 'nfr', 'B'), sp)
        image_list = os.listdir(image_fold_A)

        image_fold_AB = os.path.join(data_dir, 'nfr', 'AB', sp)
        if not os.path.isdir(image_fold_AB):
            os.makedirs(image_fold_AB)

        for n in tqdm(range(len(image_list)), desc=f"[build_nfr_dataset]: Write into AB/{sp}"):
            name_A = image_list[n]
            path_A = os.path.join(image_fold_A, name_A)

            name_B = name_A
            path_B = os.path.join(image_fold_B, name_B)

            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A
                path_AB = os.path.join(image_fold_AB, name_AB)
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
    # done flag
    with open(done_flag, "w") as fp:
        fp.write("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=['prepare', 'visualize_reconstruction', 'build_nfr_dataset'])
    parser.add_argument("--dataset_dir", type=str, default=f"{ROOT}/data/vocaset_video")
    parser.add_argument("--source_dir", type=str, default="~/assets/vocaset/Data/videos_lmks")
    parser.add_argument("--dest_size", type=int, default=256)
    parser.add_argument('--matlab_data_path', type=str, default='renderer/data/data.mat')
    parser.add_argument("--speakers", type=str, nargs="+", default=["FaceTalk_170725_00137_TA"])
    args = parser.parse_args()

    if args.mode == "prepare":
        for spk in args.speakers:
            prepare_vocaset_video(args.dataset_dir, args.source_dir, spk, dest_size=args.dest_size, training=True)
            prepare_vocaset_video(args.dataset_dir, args.source_dir, spk, dest_size=args.dest_size, training=False)
    elif args.mode == "visualize_reconstruction":
        for spk in args.speakers:
            visualize_reconstruction(args.dataset_dir, spk)
    elif args.mode == "build_nfr_dataset":
        mouth_mask = networks.MouthMask(args)
        for spk in args.speakers:
            build_nfr_dataset(mouth_mask, args.dataset_dir, spk)
