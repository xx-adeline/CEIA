import json
import h5py
import random
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
from data.utils import _convert_image_to_rgb, center_event, split_event, make_event_histogram
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class ASLDVSDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super(ASLDVSDataset, self).__init__()

        self.cfg = cfg
        self.mode = mode
        
        if mode == 'val' or mode == 'test':
            self.H5Dataset = h5py.File(self.cfg.val_path, 'r')

            with open(self.cfg.test_list, mode='r') as f:
                self.map_file = json.load(f)
            
            self.dataset_name = self.cfg.val_dataset
        else:
            raise AttributeError('ASL-DVS can only use test or val mode.')
        
        # get class names
        self.class_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 
                          'L': 10, 'M': 11, 'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18, 
                          'U': 19, 'V': 20, 'W': 21, 'X': 22, 'Y': 23}
        self.class_num = len(self.class_map)
        
        # Gray or R-B
        self.colorization_type = getattr(cfg, 'colorization_type', None)
        if self.colorization_type == 'to_rgb':
            self.red = np.array([255, 0, 0])
            self.blue = np.array([0, 0, 255])
        elif self.colorization_type == 'to_gray':
            self.red = np.array([127, 127, 127])
            self.blue = np.array([127, 127, 127])

        np.seterr(divide='ignore',invalid='ignore')

        self.Image_tran = ToPILImage()
        self.clip_transform = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])


    def __getitem__(self, idx):
        event_path = self.map_file[idx]
        event = np.array(self.H5Dataset[event_path])
        event = event.astype(np.float32)
        event[:, 2] /= 1e6                          # us, 1/0 -> s, -1/1
        event[:, 3][event[:, 3] == 0] = -1

        # Center shift event
        event = center_event(event, resolution=(180, 240))
        
        # Split event
        event = split_event(event, self.cfg.length)

        # event to hist
        event = make_event_histogram(event, (180, 240), self.red, self.blue, background_mask=self.cfg.background_mask, count_non_zero=self.cfg.count_non_zero)

        event = self.Image_tran(event)
        
        # debug input
        if self.cfg.debug_input:
            event.save('./event.png')

        # Transform event
        event = self.clip_transform(event)

        label = self.class_map[event_path.split('/')[0].upper()]

        return event, label

    def __len__(self):
        return len(self.map_file)


