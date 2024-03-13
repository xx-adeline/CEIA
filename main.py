import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from model.model_interface import MInterface
from data.data_interface import DInterface
import parse_utils


def main(cfg):
    pl.seed_everything(cfg.seed)

    # Creat Datasets
    Data = DInterface(cfg)

    # Creat Model
    model = MInterface(cfg, Data.dataset_info)

    # callbacks
    if cfg.mode == 'train':
        callbacks_list = []
        callbacks_list.append(plc.EarlyStopping(
            monitor='acc_1',
            mode='max',
            patience=cfg.earlystopping_patience,
            min_delta=0.001,
        ))
        callbacks_list.append(plc.ModelCheckpoint(
            dirpath=cfg.save_root,
            monitor='acc_1',
            filename='Best_{epoch:02d}_{acc_1:.4f}_{acc_5:.4f}',
            save_top_k=1,
            mode='max',
            save_last=False,
            every_n_epochs=1,
        ))
        my_enable_checkpointing = True
        my_logger=None
    else:
        callbacks_list = None
        my_enable_checkpointing = False
        my_logger=False

    trainer = Trainer(
        max_epochs = cfg.epochs,
        default_root_dir = cfg.save_root,
        accelerator = 'gpu',
        devices=cfg.devices,
        strategy=cfg.strategy,
        precision = cfg.mixed_precision,
        # devices=[3],
        # strategy='auto',
        # precision='32-true',
        accumulate_grad_batches = cfg.accumulate_grad_batches,
        # num_sanity_val_steps = 2,
        # limit_train_batches = 1,
        # limit_val_batches = 1,
        callbacks = callbacks_list,
        enable_checkpointing=my_enable_checkpointing,
        logger=my_logger,
    )

    if cfg.mode == 'train':
        trainer.fit(model=model, train_dataloaders=Data.train_loader, val_dataloaders=Data.val_loader)
    elif cfg.mode == 'test':
        trainer.test(model=model, dataloaders=Data.test_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='config/train.ini')
    parser.add_argument('--override', default=None)
    parser.add_argument("--local_rank", type=int)                                 
    
    args = parser.parse_args()

    # load ini
    cfg, print_format = parse_utils.parse_ini(args.config)

    # override
    if args.override is not None:
        cfg = parse_utils.override_cfg(args.override, cfg)

    # print cfg
    parse_utils.print_cfg(cfg, print_format)

    if cfg.mode == 'train':
        # save cfg
        parse_utils.save_cfg(cfg, print_format)
    
    torch.set_float32_matmul_precision("medium")

    main(cfg)


