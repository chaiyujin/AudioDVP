import os
import torch

from utils import util
from datasets.base_dataset import BaseDataset


class MultiAudioExprDataset(BaseDataset):
    """ 'AudioExpressionDataaset' requires to be TEMPORALLY CONTIGUOUS. So does this """

    def __init__(self, opt):
        super().__init__(self, opt)
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
        filename = os.path.basename(clip_data['filenames'][index])

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

        def _load_dir(clip_dir):
            ret = dict()
            ret['feature_list'] = util.load_coef(os.path.join(clip_dir, 'feature'))
            ret['filenames'] = util.get_file_list(os.path.join(clip_dir, 'feature'))

            if self.opt.isTrain:
                ret['alpha_list'] = util.load_coef(os.path.join(clip_dir, 'alpha'))
                ret['beta_list'] = util.load_coef(os.path.join(clip_dir, 'beta'))
                ret['delta_list'] = util.load_coef(os.path.join(clip_dir, 'delta'))
                ret['gamma_list'] = util.load_coef(os.path.join(clip_dir, 'gamma'))
                ret['rotation_list'] = util.load_coef(os.path.join(clip_dir, 'rotation'))
                ret['translation_list'] = util.load_coef(os.path.join(clip_dir, 'translation'))

            self.clips.append(ret)
            # append coordinates
            idx_clip = len(self.clips) - 1
            for idx_feat in range(len(ret['feature_list'])):
                self.coordinates.append((idx_clip, idx_feat))
        
        # find clips
        clip_dirs = []
        for dirpath, subdirs, _ in os.walk(self.opt.data_dir):
            for subdir in subdirs:
                if subdir.startswith("clip"):
                    clip_dirs.append(os.path.join(dirpath, subdir))
        
        # load clips
        for clip_dir in clip_dirs:
            _load_dir(clip_dir)
