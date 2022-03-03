import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from torchvision import utils

from renderer.face_model import FaceModel
from options.options import Options
from utils.util import create_dir, load_coef
from yuki11.utils import mesh_viewer, VideoWriter

mesh_viewer.set_template("template.obj")
mesh_viewer.set_shading_mode("smooth")


if __name__ == '__main__':
    opt = Options().parse_args()

    recons_dir = os.path.join(opt.spk_dir, "reconstructed", "train")
    dirpaths = []
    for cur_root, subdirs, _ in os.walk(recons_dir):
        for subdir in subdirs:
            if subdir.startswith("clip-"):
                dirpath = os.path.join(cur_root, subdir)
                dirpaths.append(dirpath)
        break

    all_alpha = []
    all_beta = []
    all_gamma = []
    all_angle = []
    all_trnsl = []
    for dirpath in dirpaths:
        alpha_list = load_coef(os.path.join(dirpath, 'alpha'      ), verbose=False)
        beta_list  = load_coef(os.path.join(dirpath, 'beta'       ), verbose=False)
        gamma_list = load_coef(os.path.join(dirpath, 'gamma'      ), verbose=False)
        angle_list = load_coef(os.path.join(dirpath, 'rotation'   ), verbose=False)
        trnsl_list = load_coef(os.path.join(dirpath, 'translation'), verbose=False)
        all_alpha.extend(alpha_list)
        all_beta .extend(beta_list)
        all_gamma.extend(gamma_list)
        all_angle.extend(angle_list)
        all_trnsl.extend(trnsl_list)
    
    alpha = torch.stack(all_alpha).mean(0, keepdim=True).cuda()
    beta  = torch.stack(all_beta ).mean(0, keepdim=True).cuda()
    gamma = torch.stack(all_gamma).mean(0, keepdim=True).cuda()
    angle = torch.stack(all_angle).mean(0, keepdim=True).cuda()
    trnsl = torch.stack(all_trnsl).mean(0, keepdim=True).cuda()

    # from predicted
    delta_list = load_coef(os.path.join(opt.src_dir, 'reenact_delta'), opt.test_num, verbose=False)

    # from target video (reconstructed)

    face_model = FaceModel(data_path=opt.matlab_data_path, batch_size=1)
    create_dir(os.path.join(opt.src_dir, 'gen3d'))

    writer = VideoWriter(os.path.join(opt.src_dir, "render.mp4"), fps=25, src_audio_path=opt.apath, high_quality=True)

    for i in tqdm(range(len(delta_list))):
        # predicted
        delta = delta_list[i].unsqueeze(0).cuda()

        geo, _ = face_model.build_face_model(alpha, delta, beta)
        verts = geo[0].detach().cpu().numpy() * 0.1
        im = mesh_viewer.render_verts(verts)[:, :, [2, 1, 0]]
        writer.write(im)

        # render
        render, _, _ = face_model(alpha, delta, beta, angle, trnsl, gamma, lower=opt.lower)
        utils.save_image(render, os.path.join(opt.src_dir, 'gen3d', '%05d.png' % (i+1)))

        if i >= opt.test_num:
            break
    writer.release()
