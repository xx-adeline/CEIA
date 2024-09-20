#!/bin/bash

for i in $*
do
    echo start to unzip part_$i.zip
    unzip -d /data/whxu/Dataset/N_Imagenet/extracted_train/  /data/whxu/Dataset/N_Imagenet/Part${i}.zip
    for f in $( ls /data/whxu/Dataset/N_Imagenet/extracted_train/Part_${i} )    
    do 
        tar -xvf /data/whxu/Dataset/N_Imagenet/extracted_train/Part_${i}/$f -C /data/whxu/Dataset/N_Imagenet/extracted_train/Part_${i}/
    done
    rm -rf /data/whxu/Dataset/N_Imagenet/extracted_train/Part_${i}/*.tar.gz

    echo start to convert part_$i 数据
    python /data/whxu/Dataset/N_Imagenet/tool/img2h5.py --save_path '/data/whxu/Dataset/N_Imagenet/extracted_train/train_set_'${i}'.hdf5' --part_path '/data/whxu/Dataset/N_Imagenet/extracted_train/Part_'${i}'/'
    rm -rf /data/whxu/Dataset/N_Imagenet/extracted_train/Part_${i}
done

