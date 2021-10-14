import os
from tqdm import tqdm
from torchvision import utils

from renderer.face_model import FaceModel
from options.options import Options
from utils.util import create_dir, load_coef


if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'reenact'))

    # from predicted
    delta_list = load_coef(os.path.join(opt.src_dir, 'reenact_delta'), opt.test_num, verbose=False)

    # from target video (reconstructed)
    alpha_list = load_coef(os.path.join(opt.tgt_dir, 'reconstructed', 'alpha'      ), opt.test_num, verbose=False)
    beta_list  = load_coef(os.path.join(opt.tgt_dir, 'reconstructed', 'beta'       ), opt.test_num, verbose=False)
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'reconstructed', 'gamma'      ), opt.test_num, verbose=False)
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'reconstructed', 'rotation'   ), opt.test_num, verbose=False)
    trnsl_list = load_coef(os.path.join(opt.tgt_dir, 'reconstructed', 'translation'), opt.test_num, verbose=False)

    face_model = FaceModel(data_path=opt.matlab_data_path, batch_size=1)

    for i in tqdm(range(len(delta_list))):
        # predicted
        delta = delta_list[i].unsqueeze(0).cuda()
        # target video (reconstructed)
        alpha = alpha_list[i + opt.offset].unsqueeze(0).cuda()
        beta  = beta_list [i + opt.offset].unsqueeze(0).cuda()
        gamma = gamma_list[i + opt.offset].unsqueeze(0).cuda()
        rotat = angle_list[i + opt.offset].unsqueeze(0).cuda()
        trnsl = trnsl_list[i + opt.offset].unsqueeze(0).cuda()

        # render
        render, _, _ = face_model(alpha, delta, beta, rotat, trnsl, gamma, lower=True)
        utils.save_image(render, os.path.join(opt.src_dir, 'reenact', '%05d.png' % (i+1)))

        if i >= opt.test_num:
            break
