import h5py
from pathlib import Path
from tqdm import tqdm


h5_path_list = [
                '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_1.hdf5',
                '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_2.hdf5',
                '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_3.hdf5',
                '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_4.hdf5',
                '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_5.hdf5',
                ]

out_path = '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set.hdf5'


Outdataset = h5py.File(out_path, 'w')

for h5_path in h5_path_list:
    with h5py.File(h5_path, mode='r') as Oridataset:

        def copy_data(name):
            if len(name.split('/')) == 2:
                Outdataset.create_dataset(name, data=Oridataset[name])
                pbar.update(1)

        with tqdm(total=150000) as pbar:
            pbar.set_description('Processing:' + str(Path(h5_path).stem))
            Oridataset.visit(copy_data)

Outdataset.close()


Outdataset = h5py.File(out_path, 'r')
file_name = []
def get_name(name):
    if len(name.split('/')) == 2:
        file_name.append(name)
Outdataset.visit(get_name)
print(len(file_name))
Outdataset.close()