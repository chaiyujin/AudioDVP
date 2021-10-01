import os
import torch

from utils import util
from datasets.base_dataset import BaseDataset


class MultiAudioExprDataset(BaseDataset):
    """ 'AudioExpressionDataaset' requires to be TEMPORALLY CONTIGUOUS. So does this """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.load_data()

    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, index):
        idx_clip, idx_feat = self.coordinates[index]
        return self.get_index_from_clip(self.clips[idx_clip], idx_feat)

    def get_index_from_clip(self, clip_data, index):
        Nw = self.opt.Nw
        max_frames = len(clip_data['feature_list'])

        # window of features
        feature_list = []
        if index < Nw // 2:
            for i in range(Nw // 2 - index):
                feature_list.append(torch.zeros(256, dtype=torch.float32))

            for i in range(index + Nw // 2 + 1):
                feature_list.append(clip_data['feature_list'][i])
        elif index > max_frames - Nw // 2 - 1:
            for i in range(index - Nw // 2, max_frames):
                feature_list.append(clip_data['feature_list'][i])

            for i in range(index + Nw // 2 - max_frames + 1):
                feature_list.append(torch.zeros(256, dtype=torch.float32))
        else:
            for i in range(index - Nw // 2, index + Nw // 2 + 1):
                feature_list.append(clip_data['feature_list'][i])
        # stack
        feature = torch.stack(feature_list, dim=0)
        # filename
        filename = clip_data['filenames'][index]

        if not self.opt.isTrain:
            return {'feature': feature, 'filename': filename}
        else:
            alpha       = clip_data['alpha_list'][index]
            beta        = clip_data['beta_list'][index]
            delta       = clip_data['delta_list'][index]
            gamma       = clip_data['gamma_list'][index]
            rotation    = clip_data['rotation_list'][index]
            translation = clip_data['translation_list'][index]

            return {
                'feature': feature, 'filename': filename,
                'alpha': alpha, 'beta': beta, 'delta_gt': delta,
                'gamma': gamma, 'translation': translation, 'rotation': rotation
            }

    def load_data(self):
        self.clips = list()
        self.coordinates = list()

        self.speaker = None

        def _load_dir(clip_dir):
            # check only one speaker
            spk = os.path.basename(os.path.dirname(os.path.dirname(clip_dir)))
            if self.speaker is not None:
                assert spk == self.speaker, "Multiple speakers!"
            else:
                self.speaker = spk

            ret = dict()
            ret['feature_list'] = util.load_coef(os.path.join(clip_dir, 'feature'), verbose=False)
            ret['filenames']    = util.get_file_list(os.path.join(clip_dir, 'feature'))

            if self.opt.isTrain:
                ret['alpha_list']       = util.load_coef(os.path.join(clip_dir, 'alpha'      ), verbose=False)
                ret['beta_list']        = util.load_coef(os.path.join(clip_dir, 'beta'       ), verbose=False)
                ret['delta_list']       = util.load_coef(os.path.join(clip_dir, 'delta'      ), verbose=False)
                ret['gamma_list']       = util.load_coef(os.path.join(clip_dir, 'gamma'      ), verbose=False)
                ret['rotation_list']    = util.load_coef(os.path.join(clip_dir, 'rotation'   ), verbose=False)
                ret['translation_list'] = util.load_coef(os.path.join(clip_dir, 'translation'), verbose=False)

            self.clips.append(ret)
            # append coordinates
            idx_clip = len(self.clips) - 1
            for idx_feat in range(len(ret['feature_list'])):
                self.coordinates.append((idx_clip, idx_feat))
        
        # find clips
        is_train = self.opt.isTrain  # load correct part
        clip_dirs = util.find_clip_dirs(self.opt.data_dir, with_train=is_train, with_test=(not is_train))
 
        # load clips
        for clip_dir in clip_dirs:
            # print(clip_dir)
            _load_dir(clip_dir)

        print("MultiAudioExprDataset has loaded data for speaker: {}".format(self.speaker))
