import os
import re
import sys
import cv2
import toml
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from models import networks
from utils import util

ROOT = os.path.dirname(os.path.abspath(__file__))


def calc_bbox(lmks_list):
    """Batch infer of face location, batch_size should be factor of total frame number."""
    x_sum, y_sum, w_sum, h_sum = 0, 0, 0, 0

    for lmks in lmks_list:
        x = lmks[..., 0].min()
        y = lmks[..., 1].min()
        w = lmks[..., 0].max() - x
        h = lmks[..., 1].max() - y
        x_sum += x
        y_sum += y
        w_sum += w
        h_sum += h
    
    x = x_sum / len(lmks_list)
    y = y_sum / len(lmks_list)
    w = w_sum / len(lmks_list)
    h = h_sum / len(lmks_list)

    cx = x + w // 2
    cy = y + h // 2
    a = int(max(w, h) * 1.5 // 2)
    x = int(max(cx - a, 0))
    y = int(max(cy - a, 0))
    w = a * 2
    h = a * 2

    return x, y, w, h


def prepare_vocaset(output_root, data_root, training, dest_size=256, debug=False):
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, "train" if training else "test")

    data_root = os.path.expanduser(data_root)

    seq_range = list(range(0, 20)) if training else list(range(20, 40))
    for i_seq in tqdm(seq_range, desc=f"[prepare_vocaset]: {os.path.basename(data_root)}/{seq_range[0]}-{seq_range[-1]}"):
        # data source
        prefix = os.path.join(data_root, f"sentence{i_seq+1:02d}")
        vpath = prefix + ".mp4"
        lpath = prefix + "-lmks-ibug-68.toml"
        # output dir
        out_dir = os.path.join(output_root, f"clip-sentence{i_seq+1:02d}")
        # preprocess
        _preprocess_video(out_dir, vpath, lpath, dest_size, debug)


def prepare_celebtalk(output_root, data_root, training, dest_size=256, debug=False, use_seqs=""):
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, "train" if training else "test")

    data_root = os.path.expanduser(data_root)

    tasks = []
    for cur_root, _, files in os.walk(data_root):
        for fpath in files:
            seq_id = os.path.splitext(fpath)[0].replace("-fps25", "")
            if seq_id not in use_seqs:
                continue
            if training:
                if re.match(r"trn-\d+-fps25\.mp4", fpath) is not None:
                    tasks.append(os.path.join(cur_root, fpath))
            else:
                if re.match(r"vld-\d+-fps25\.mp4", fpath) is not None:
                    tasks.append(os.path.join(cur_root, fpath))
        break
    print(tasks)

    for vpath in tqdm(tasks, desc=f"[prepare_celebtalk]: {os.path.basename(data_root)}"):
        # data source
        lpath = os.path.splitext(vpath)[0] + "-lmks-ibug-68.toml"
        assert os.path.exists(lpath)
        # output dir
        seq_id = os.path.basename(os.path.splitext(vpath)[0]).replace("-fps25", "")
        out_dir = os.path.join(output_root, f"clip-{seq_id}")
        # preprocess
        _preprocess_video(out_dir, vpath, lpath, dest_size, debug)
       

def _preprocess_video(out_dir, vpath, lpath, dest_size, debug):
    # skip if already done before
    if os.path.exists(os.path.join(out_dir, "landmark.pkl")):
        return

    # prepare output dirs
    os.makedirs(os.path.join(out_dir, "full"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crop"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "feature"), exist_ok=True)

    # 1. -> 25 fps images and audio
    # assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} -r 25 {out_dir}/full/%05d.png") == 0
    assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} {out_dir}/full/%05d.png") == 0
    assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} {out_dir}/audio/audio.wav") == 0

    # 2. audio fetures
    assert os.system(f"{sys.executable} {ROOT}/vendor/ATVGnet/code/test.py -i {out_dir}/") == 0

    # 3. resize and dump landmarks
    with open(lpath) as fp:
        lmks_data = toml.load(fp)
        assert lmks_data["fps"] == 25
    lmks_mapping = dict()

    # # find bbox first
    # all_lmks = np.asarray([x['points'] for x in lmks_data["frames"]], dtype=np.float32)
    # x, y, w, h = calc_bbox(all_lmks)

    img_list = sorted(glob(f"{out_dir}/full/*.png"))
    for i_frm, img_path in enumerate(img_list):
        save_path = f"{out_dir}/crop/{os.path.basename(img_path)}"
        img = cv2.imread(img_path)
        pts = np.asarray(lmks_data['frames'][i_frm]['points'], dtype=np.float32)

        # resize
        pts[:, 0] = pts[:, 0] / img.shape[1] * dest_size
        pts[:, 1] = pts[:, 1] / img.shape[0] * dest_size
        img = cv2.resize(img, (dest_size, dest_size))
        # update mapping
        lmks_mapping[save_path] = pts
        cv2.imwrite(save_path, img)
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


def visualize_reconstruction(data_dir, recons_dir):
    # find clips
    clip_dirs = util.find_clip_dirs(data_dir, with_train=True, with_test=True)

    for clip_dir in tqdm(clip_dirs, desc="[visualize_reconstruction]"):
        ss = clip_dir.split('/')
        clip_recons_dir = os.path.join(recons_dir, ss[-2], ss[-1])
        if os.path.exists(clip_recons_dir + "-debug.mp4"):
            continue
        cmd = (
            "ffmpeg -hide_banner -n -loglevel error "
            "-thread_queue_size 8192 -i {1}/render/%05d.png "
            "-thread_queue_size 8192 -i {1}/overlay/%05d.png "
            "-thread_queue_size 8192 -i {0}/crop/%05d.png "
            "-i {0}/audio/audio.wav "
            "-filter_complex hstack=inputs=3 -vcodec libx264 -preset slower "
            "-profile:v high -crf 18 -pix_fmt yuv420p {1}-debug.mp4"
        ).format(clip_dir, clip_recons_dir)
        assert os.system(cmd) == 0


def generate_masks(mouth_mask, recons_dir, debug):
    # find training clips
    clip_dirs = util.find_clip_dirs(recons_dir, with_train=True, with_test=True)

    # generate mask for all clips
    for clip_dir in tqdm(clip_dirs, desc="[generate_masks]: Generate masks"):
        done_flag = os.path.join(clip_dir, 'done.mask')
        if os.path.exists(done_flag):
            continue

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

            save_path = os.path.join(clip_dir, 'mask', '%05d.png' % (i+1))
            cv2.imwrite(save_path, mask)
            if debug:
                cv2.imshow('mask', mask)
                cv2.waitKey(1)
        
        with open(done_flag, "w") as fp:
            fp.write("")


def build_nfr_dataset(data_dir, recons_dir, nfr_data_dir):
    # done lock
    done_flag = os.path.join(nfr_data_dir, "done.flag")
    if os.path.exists(done_flag):
        print("[build_nfr_dataset]: Find {}. It's already done! Skip.".format(done_flag))
        return

    # find training clips
    # !IMPORTANT: Only training
    clip_dirs = util.find_clip_dirs(data_dir, with_train=True, with_test=False)

    # collect all masks, crops and renders from clips
    masks, crops, renders = [], [], []
    for clip_dir in clip_dirs:
        ss = clip_dir.split('/')
        clip_recons_dir = os.path.join(recons_dir, ss[-2], ss[-1])
        crops  .extend(util.get_file_list(os.path.join(clip_dir, 'crop')))
        masks  .extend(util.get_file_list(os.path.join(clip_recons_dir, 'mask')))
        renders.extend(util.get_file_list(os.path.join(clip_recons_dir, 'render')))

    # create dir for nfr dataset
    util.create_dir(os.path.join(nfr_data_dir, 'A', 'train'))
    util.create_dir(os.path.join(nfr_data_dir, 'B', 'train'))
    # save into nfr dataset
    for i in tqdm(range(len(masks)), desc="[build_nfr_dataset]: Write into A/train or B/train"):
        mask   = cv2.imread(masks[i])
        crop   = cv2.imread(crops[i])
        render = cv2.imread(renders[i])

        masked_crop   = cv2.bitwise_and(crop, mask)
        masked_render = cv2.bitwise_and(render, mask)

        cv2.imwrite(os.path.join(nfr_data_dir, 'A', 'train', '%05d.png' % (i+1)), masked_crop)
        cv2.imwrite(os.path.join(nfr_data_dir, 'B', 'train', '%05d.png' % (i+1)), masked_render)

    # create AB dataset from A, B dirs
    for sp in os.listdir(os.path.join(nfr_data_dir, 'A')):
        image_fold_A = os.path.join(os.path.join(nfr_data_dir, 'A'), sp)
        image_fold_B = os.path.join(os.path.join(nfr_data_dir, 'B'), sp)
        image_list = os.listdir(image_fold_A)

        image_fold_AB = os.path.join(nfr_data_dir, 'AB', sp)
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
                im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)

    # done flag
    with open(done_flag, "w") as fp:
        fp.write("")


if __name__ == "__main__":
    import argparse
    choices = ['prepare_vocaset', 'prepare_celebtalk', 'visualize_reconstruction', 'generate_masks', 'build_nfr_dataset']

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=choices)
    parser.add_argument("--vocaset_dir", type=str, default="~/assets/vocaset")
    parser.add_argument("--celebtalk_dir", type=str, default="~/assets/CelebTalk")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--recons_dir", type=str, default=None)
    parser.add_argument("--nfr_data_dir", type=str, default=None)
    parser.add_argument("--use_seqs", type=str, default="")
    parser.add_argument("--dest_size", type=int, default=256)
    parser.add_argument('--matlab_data_path', type=str, default='renderer/data/data.mat')
    parser.add_argument("--lower", action="store_true", help="only use lower face")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mode == "prepare_vocaset":
        spk_dir = os.path.join(args.vocaset_dir, "Data", "videos_lmks_crop", args.speaker)
        prepare_vocaset(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, training=True)
        prepare_vocaset(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, training=False)
    elif args.mode == "prepare_celebtalk":
        spk_dir = os.path.join(args.celebtalk_dir, "ProcessTasks", args.speaker, "clips_cropped")
        prepare_celebtalk(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, use_seqs=args.use_seqs, training=True)
        prepare_celebtalk(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, use_seqs=args.use_seqs, training=False)
    elif args.mode == "visualize_reconstruction":
        visualize_reconstruction(args.data_dir, args.recons_dir)
    elif args.mode == "generate_masks":
        mouth_mask = networks.MouthMask(args)
        generate_masks(mouth_mask, args.recons_dir, debug=args.debug)
    elif args.mode == "build_nfr_dataset":
        build_nfr_dataset(args.data_dir, args.recons_dir, args.nfr_data_dir)
