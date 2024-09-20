import h5py
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def img2h5(cfg):

    # Verify that labels are consistent
    labels = [s.split()[1].strip() for s in open(cfg.mapping_path, 'r').readlines()]
    labels = sorted(labels[:1000])        # N_imageNet仅用1000类

    part_labels = sorted([p.name for p in Path(cfg.part_path).iterdir()])
    old_len = len(part_labels)
    part_labels = list(filter(lambda p: p in labels, part_labels))
    print('the number of classes is {}'.format(len(part_labels)))
    assert len(part_labels) == old_len, 'Label inconsistency.'
    
    
    # Recording Path
    map_file = []
    for label_path in sorted(Path(cfg.part_path).iterdir(), key=lambda x: x.stem):
        for instance_path in sorted(label_path.iterdir(), key=lambda x: int(x.stem.split('_')[1])):
            map_file.append(instance_path)

    # Start Conversion
    H5File = h5py.File(cfg.save_path, 'w')

    with tqdm(map_file, desc='img to h5') as tbar:
        for event_path in tbar:
            event = event = np.load(event_path)['event_data']
            H5File.create_dataset(event_path.parent.name + '/' + event_path.stem, data=event)

    H5File.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_6.hdf5')
    parser.add_argument('--mapping_path', default='/data/whxu/Dataset/N_Imagenet/mapping.txt')
    parser.add_argument('--part_path', default='/data/whxu/Dataset/N_Imagenet/extracted_train/Part_6/')

    cfg = parser.parse_args()

    img2h5(cfg)
