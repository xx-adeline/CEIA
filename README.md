# CEIA: CLIP-Based Event-Image Alignment for Open-World Event-Based Understanding

<img src="images\CEIA.jpg"  />

Official implementation of the following paper:

CEIA: CLIP-Based Event-Image Alignment for Open-World Event-Based Understanding by Wenhao Xu, Wenming Weng, Yueyi Zhang, and Zhiwei Xiong. In ECCV workshop 2024.

## Dataset: 
See the files in the `Data\misc` folder for the dataset structure.

| Event Datasets | Access to Download | Corresponding Image Datasets | Access to Download |
|-----------------|--------------------|-------------------------------|---------------------|
| N-ImageNet       | [Download](https://github.com/82magnolia/n_imagenet) | ImageNet | [Download](https://image-net.org) |
| N-Caltech101     | [Download](https://github.com/uzh-rpg/rpg_event_representation_learning) | Caltech101 | [Download](https://data.caltech.edu/records/mzrjq-6wc02) |
| CIFAR10-DVS      | [Download](https://figshare.com/s/d03a91081824536f12a8) |  -  |  -  |
| ASL-DVS          | [Download](https://github.com/PIX2NVS/NVS2Graph) |  -  |  -  |

Our built event-text dataset "NIN-BLIP2" and "NIN-BLIP2-retrieval" are accessible at the path `event-text-dataset`

## Pretrained model
The pre-trained checkpoint is located at the path `output\pretrain\Best.ckpt`

## Acknowledgement
We thank the authors of [EventCLIP](https://github.com/Wuziyi616/EventCLIP), [N-ImageNet](https://github.com/82magnolia/n_imagenet), and [LoRA](https://github.com/Baijiong-Lin/LoRA-Torch) for opening source their wonderful works.

## Citation
If you find this work helpful, please consider citing our paper.

```
@article{xu2024ceia,
  title={CEIA: CLIP-Based Event-Image Alignment for Open-World Event-Based Understanding},
  author={Xu, Wenhao and Weng, Wenming and Zhang, Yueyi and Xiong, Zhiwei},
  journal={arXiv preprint arXiv:2407.06611},
  year={2024}
}
```


## Contact
If you have any problem about the released code, please do not hesitate to contact me with email (wh-xu@mail.ustc.edu.cn).