import os
import pickle
from PIL import Image
import torch
from torchvision import transforms

from utils import util
from datasets.base_dataset import BaseDataset


class MultiDataset(BaseDataset):
    """ 'SingleDataset' doesn't require to be temporally contiguous. So does MultiDataset """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # load data paths
        self.load_images_and_landmarks()

        # transforms
        self.transforms_input = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5141, 0.4074, 0.3588], std=[1.0, 1.0, 1.0])
        ])
        self.transforms_gt = transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(image_name).convert('RGB')

        input = self.transforms_input(image)
        gt = self.transforms_gt(image)
        landmark_gt = torch.tensor(self.landmark_dict[image_name])

        return {
            'input': input,
            'gt': gt,
            'landmark_gt': landmark_gt,
            'image_name': image_name,
        }

    def load_images_and_landmarks(self):
        # prepare list and dict
        self.image_list = list()
        self.landmark_dict = dict()

        # find clips
        # !IMPORTANT: Load both train and test data for 3D reconstruction!
        clip_dirs = util.find_clip_dirs(self.opt.data_dir, with_train=True, with_test=True)
        
        # load from each clip
        for clip_dir in clip_dirs:
            img_list = util.get_file_list(os.path.join(clip_dir, 'crop'))
            lmk_path = os.path.join(clip_dir, 'landmark.pkl')
            if not os.path.exists(lmk_path):
                util.landmark_detection(img_list, lmk_path)
            with open(lmk_path, 'rb') as f:
                lmk_dict = pickle.load(f)
            # extend
            self.image_list.extend(img_list)
            self.landmark_dict.update(lmk_dict)
