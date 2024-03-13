import torch
from clip.clip import tokenize
import torchmetrics
import pytorch_lightning as pl
from timm.scheduler.cosine_lr import CosineLRScheduler
from model.elip import ELIP
import loratorch as lora


class MInterface(pl.LightningModule):
    def __init__(self, cfg, dataset_info):
        super().__init__()
        print(f'Initializing model {cfg.vit}...')
        self.cfg = cfg
        self.class_map = dataset_info[0]
        self.class_num = dataset_info[1]
        self.save_hyperparameters(ignore=cfg)

        self.elip = ELIP(self.cfg, self.device)

        # Metric Acc
        self.acc_1 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.class_num)
        if self.class_num > 5:
            self.acc_5 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.class_num, top_k=5)
    
    def configure_optimizers(self):
        self.print('Using AdamW as optimizer')

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)    # primary parameters

        named_parameters = list(self.elip.named_parameters())
        
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        
        optimizer = torch.optim.Adam(
                            [{"params": gain_or_bias_params, "weight_decay": 0.},
                            {"params": rest_params, "weight_decay": self.cfg.weight_decay},],
                            lr=self.cfg.lr
                            )

        self.print('Using CosineLRScheduler as scheduler')
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.cfg.epochs,              
            lr_min=self.cfg.min_lr,
            warmup_lr_init=self.cfg.min_lr,
            warmup_t=self.cfg.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.current_epoch)

    def training_step(self, batch, batch_idx):
        
        event, image = batch['event'], batch['image']
        itc_loss = self.elip(event, image)

        self.log('itc_loss', itc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        torch.cuda.empty_cache()

        return itc_loss

    def validation_step(self, batch, batch_idx):
        event, label = batch['event'], batch['label']
        event_feat = self.elip.encode_event(event)
        logits = (self.elip.clip.logit_scale.exp() * event_feat @ self.zeroshot_weight).softmax(dim=-1)

        self.acc_1(logits, label)
        self.log('acc_1', self.acc_1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if self.class_num > 5:
            self.acc_5(logits, label)
            self.log('acc_5', self.acc_5, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_fit_start(self):
        self.zeroshot_weight = self.get_zeroshot_weights()

    def on_test_start(self):
        self.zeroshot_weight = self.get_zeroshot_weights()

    def get_zeroshot_weights(self):
        # Using prompt to obtain classifier weights

        if self.cfg.val_dataset == 'N_Caltech':
            template = 'image of a {}.'
        elif self.cfg.val_dataset == 'N_Mnist':
            template = 'a point cloud image of the number: "{}".'
        elif self.cfg.val_dataset == 'ASL_DVS':
            template = 'a point cloud image representing the American sign language letter "{}".'
        elif self.cfg.val_dataset == 'DVS_Gesture':
            template = 'a point cloud image capturing the gesture: "{}".'
        else:
            template = 'a point cloud image of a {}.'

        if self.cfg.val_dataset == 'N_ImageNet':
            texts = [template.format(classname[0]) for classname in self.class_map.values()]
        elif self.cfg.val_dataset == 'ASL_DVS':
            texts = [template.format(classname) for classname in self.class_map.keys()]            
        else:
            texts = [template.format(classname) for classname in self.class_map]

        with torch.no_grad():           
            texts = tokenize(texts).to(self.device)                                         # 100*77
            class_embeddings = self.elip.clip.encode_text(texts)                            # 100*768
            class_embeddings /= (class_embeddings.norm(dim=-1, keepdim=True) + 1e-9)        # 100*768
            zeroshot_weights = class_embeddings.permute(1, 0)                               # 768*100

        return zeroshot_weights
    
    def on_save_checkpoint(self, checkpoint):
        # Only save LoRA
        if self.cfg.lora:
            del checkpoint['state_dict']
            checkpoint['event_encoder'] = lora.lora_state_dict(self.elip.event_encoder, bias='all')
            checkpoint['event_logit_scale'] = self.elip.event_logit_scale