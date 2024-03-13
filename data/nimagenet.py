import json
import h5py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
from data.utils import _convert_image_to_rgb, parse_event, center_event, split_event, make_event_histogram, event_augment
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class NimageNetDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super(NimageNetDataset, self).__init__()

        self.cfg = cfg
        self.mode = mode
        
        if mode == 'train':
            self.image_file = Path(self.cfg.image_path)
            # load train_set.hdf5
            self.H5Dataset = h5py.File(self.cfg.train_path, 'r')

            # load train event_path
            with open(self.cfg.train_list, mode='r') as f:
                self.map_file = json.load(f)

        elif mode == 'val' or mode == 'test':
            # load val_set.hdf5
            self.H5Dataset = h5py.File(self.cfg.val_path, 'r')

            # load test event_path
            with open(self.cfg.test_list, mode='r') as f:
                self.map_file = json.load(f)

        # get class names {'n0211978':'kit fox', ...}
        self.class_map = {s.split(maxsplit=1)[0].strip():s.split(maxsplit=1)[1].strip() for s in open(cfg.classes_path, 'r').readlines()}

        # get labels [n02119789, ...]
        self.labels = [s.split()[1].strip() for s in open(cfg.labels_path, 'r').readlines()]
        self.labels = sorted(self.labels[:1000])        # 1000 classes

        if mode == 'train':
            if cfg.train_dataset == 'N_ImageNet_Half':
                self.labels = self.labels[:500]         # Large, the first 500 classes for training
            elif cfg.train_dataset == 'N_ImageNet_Mini':
                self.labels = self.labels[:100]         # Small, the first 100 classes for training
            
            self.dataset_name = cfg.train_dataset

        elif mode == 'val' or mode == 'test': 
            self.labels = self.labels[500:]             # The last 500 classes for evaluation
            self.class_num = 500

            self.dataset_name = cfg.val_dataset

        # Filter out training/test classes
        self.map_file = list(filter(lambda s: s.split('/')[0] in self.labels, self.map_file))        
        self.class_map = dict(filter(lambda s: s[0] in self.labels, self.class_map.items()))          

        # {'n0211978': ['kit fox', 0], ...}
        self.class_map = {s[0]:[s[1], idx] for idx, s in enumerate(self.class_map.items())}

        # Gray or R-B
        self.colorization_type = getattr(cfg, 'colorization_type', None)
        if self.colorization_type == 'to_rgb':
            self.red = np.array([255, 0, 0])
            self.blue = np.array([0, 0, 255])
        elif self.colorization_type == 'to_gray':
            self.red = np.array([127, 127, 127])
            self.blue = np.array([127, 127, 127])

        np.seterr(divide='ignore',invalid='ignore')

        # Transform
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

        event = self.H5Dataset[event_path]
        event = parse_event(event)      # us, 1/0 --> s, -1/1

        # Center shift event
        event = center_event(event, resolution=(480, 640))

        # Split event
        event = split_event(event, 70000)

        if self.cfg.augment_event:
            event = event_augment(event, self.cfg, (480, 640))

        # event to hist
        event = make_event_histogram(event, (480, 640), self.red, self.blue, background_mask=self.cfg.background_mask, count_non_zero=self.cfg.count_non_zero)

        event = self.Image_tran(event)
        
        # debug input
        if self.cfg.debug_input:
            event.save('./event.png')

        # Transform event
        event = self.clip_transform(event)

        if self.mode == 'train':
            image_path =  (self.image_file / event_path ).with_suffix('.JPEG')
            image = Image.open(image_path).convert('RGB')

            # debug input
            if self.cfg.debug_input:
                image.save('./image.png')
                
            image = self.clip_transform(image)
            return event, image
        else:
            label = self.class_map[event_path.split('/')[0]][1]
            return event, label

    def __len__(self):
        return len(self.map_file)


