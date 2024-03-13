import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.nimagenet import NimageNetDataset
from data.ncaltech import NcaltechDataset
from data.cifar10 import CifarDvsDataset
from data.asl_dvs import ASLDVSDataset


class DInterface(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gen_dataset()
        self.gen_dataloader()

    def gen_dataset(self):
            
            if self.cfg.mode == 'train':
                # Train Datasets
                print(f'Initializing trainning Dataset {self.cfg.train_dataset}...')
                if self.cfg.train_dataset in ['N_ImageNet_Half', 'N_ImageNet_Mini']:
                    self.train_dataset = NimageNetDataset(self.cfg, mode='train')
                else:
                     raise AttributeError(f'Do not provide this training set: {self.cfg.train_dataset}')
                
                # Val Datasets
                print(f'Initializing validation Dataset {self.cfg.val_dataset}...')
                if self.cfg.val_dataset == 'N_ImageNet':
                    self.val_dataset = NimageNetDataset(self.cfg, mode='val')
                else:
                    raise AttributeError(f'Do not provide this validation set: {self.cfg.val_dataset}')
                
                self.dataset_info = [self.val_dataset.class_map, self.val_dataset.class_num]
            
            elif self.cfg.mode == 'test':
                # Test Datasets
                print(f'Initializing testing Dataset {self.cfg.val_dataset}...')
                if self.cfg.val_dataset == 'N_ImageNet':
                    self.test_dataset = NimageNetDataset(self.cfg, mode='test')
                elif self.cfg.val_dataset == 'N_Caltech':
                    self.test_dataset = NcaltechDataset(self.cfg, mode='test')
                elif self.cfg.val_dataset == 'Cifar_DVS':
                    self.test_dataset = CifarDvsDataset(self.cfg, mode='test')
                elif self.cfg.val_dataset == 'ASL_DVS':
                    self.test_dataset = ASLDVSDataset(self.cfg, mode='test')
                    raise AttributeError(f'Do not provide this testing set: {self.cfg.val_dataset}')
                
                self.dataset_info = [self.test_dataset.class_map, self.test_dataset.class_num]
            
            else:
                raise AttributeError('Mode not provided')

    def gen_dataloader(self):
        
        if self.cfg.mode == 'train':
            self.train_loader = DataLoader(self.train_dataset, collate_fn=dict_collate_fn_for_train,
                batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, drop_last=True, pin_memory=self.cfg.pin_memory)

            self.val_loader = DataLoader(self.val_dataset, collate_fn=dict_collate_fn_for_val,
                batch_size=self.cfg.batch_size_val, shuffle=False, num_workers=self.cfg.num_workers, drop_last=False, pin_memory=self.cfg.pin_memory)

        elif self.cfg.mode == 'test':
            self.test_loader = DataLoader(self.test_dataset, collate_fn=dict_collate_fn_for_val,
                batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, drop_last=False, pin_memory=self.cfg.pin_memory)

def dict_collate_fn_for_train(list_data):
    event_list, image_list = list(zip(*list_data))

    event_batch = torch.stack(event_list, dim=0)        # N, 3, H, W
    image_batch = torch.stack(image_list, dim=0)        # N, 3, H, W
    return {
            'event': event_batch,
            'image': image_batch,
            }

def dict_collate_fn_for_val(list_data):
    event_list, label_list = list(zip(*list_data))

    event_batch = torch.stack(event_list, dim=0)        # N, 3, H, W
    label_batch = torch.LongTensor(label_list)          # N
    return {
            'event': event_batch,
            'label': label_batch
            }

