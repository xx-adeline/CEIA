from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
import json
from pathlib import Path
import tonic.transforms as transforms
from tonic.io import read_aedat4
from data.utils import _convert_image_to_rgb, split_event, make_event_histogram, split_event, center_event
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class CifarDvsDataset(Dataset):
    def __init__(self, cfg, mode):
        super(CifarDvsDataset).__init__()

        self.cfg = cfg
        self.mode = mode

        self.Image_tran = ToPILImage()   
        self.clip_transform = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.denoise_transform = transforms.Compose([transforms.Denoise(filter_time=15000),])

        if mode == 'val' or mode == 'test':
            with open(self.cfg.test_list, mode='r') as f:
                self.map_file = json.load(f)

            self.val_path = Path(self.cfg.val_path)
            self.dataset_name = cfg.val_dataset
        else:
            raise AttributeError('Cifar10 can only use test or val mode.')

        # get class names
        self.class_map = sorted([cls.stem for cls in self.val_path.iterdir()])
        self.class_num = len(self.class_map)
        self.tonic_dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
        
        # get event file paths and labelss
        self.file_path_list = []
        self.label_list = []
        for file, label in self.map_file:
            self.file_path_list.append(file)
            self.label_list.append(label)
        assert len(self.file_path_list) == len(self.label_list)

        # Gray or R-B
        self.colorization_type = getattr(cfg, 'colorization_type', None)
        if self.colorization_type == 'to_rgb':
            self.red = np.array([255, 0, 0])
            self.blue = np.array([0, 0, 255])
        elif self.colorization_type == 'to_gray':
            self.red = np.array([127, 127, 127])
            self.blue = np.array([127, 127, 127])

        np.seterr(divide='ignore',invalid='ignore')


    def __getitem__(self, idx):
        label = self.label_list[idx]
        event_path = self.val_path / '/'.join(self.file_path_list[idx].split('/')[5:])
        event = read_aedat4(event_path)
        event.dtype.names = ["t", "x", "y", "p"]
        event = self.denoise_transform(event)
        
        event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T      # (n. 4)  
        event = event.astype(np.float32)
        event[:, 2] /= 1e6                          # us, 1/0 -> s, -1/1
        event[:, 3][event[:, 3] == 0] = -1

        # Center shift event
        event = center_event(event, resolution=(128, 128))  

        # Split event
        event = split_event(event, self.cfg.length)

        # event to hist
        event = make_event_histogram(event, (128, 128), self.red, self.blue, background_mask=self.cfg.background_mask, count_non_zero=self.cfg.count_non_zero)

        event = self.Image_tran(event)
        
        # debug input
        if self.cfg.debug_input:
            event.save('./event.png')

        # Transform event
        event = self.clip_transform(event)

        return event, label

    def __len__(self):
        return len(self.file_path_list)








