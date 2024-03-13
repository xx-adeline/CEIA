import torch
from clip.clip import load
import torch.nn as nn
import numpy as np
import loratorch as lora
import torch.nn.functional as F
from model.vit import VisionTransformer


class ELIP(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg

        self.event_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.clip, _ = load('ViT-L/14', device=device, download_root=self.cfg.pretrain_path)
        self.image_encoder = self.clip.visual

        # ViT-L/14
        self.event_encoder = VisionTransformer(
                                        input_resolution=224,
                                        patch_size=14,
                                        width=1024,
                                        layers=24,
                                        heads=16,
                                        output_dim=768
                                        )
        
        # Initializing the Event encoder with CLIP Image encoder
        zs_event_state_dict = self.image_encoder.state_dict()
        self.event_encoder.load_state_dict(zs_event_state_dict)

        # Whether to use LoRa
        if self.cfg.lora:
            print('Lora fine-tuning query / value...')
            r = self.cfg.r
            alpha = self.cfg.alpha

            for i in range(24):
                module_key = f'transformer.resblocks.{i}.attn'
                submodule = self.event_encoder.get_submodule(module_key)
                module_state_dict = submodule.state_dict()
                lora_attn = lora.MultiheadAttention(
                    embed_dim=1024,
                    num_heads=16,
                    enable_lora=['q', 'v'],
                    r=r, 
                    lora_alpha=alpha,
                )
                # Import pre-trained parameters
                lora_attn.load_state_dict(module_state_dict, strict=False)
                # Replace 
                self.event_encoder.transformer.resblocks[i].attn = lora_attn

            # Freeze parameters except for LoRA and bias
            lora.mark_only_lora_as_trainable(self.event_encoder, bias='all')
        else:
            print('Full fine-tuning... / Using CLIP original parameters ')

        # Freeze CLIP 
        print('Freezing CLIP...')
        for param in self.clip.parameters():
            param.requires_grad_(False)

        # Test
        if self.cfg.mode == 'test':

            # Import trained parameters
            if cfg.trained:
                print('Using trained params...')
                ckpt = torch.load(cfg.pt_path)
                self.event_encoder.load_state_dict(ckpt['event_encoder'], strict=False)
                self.event_logit_scale = nn.Parameter(ckpt['event_logit_scale'].data)                

    def forward(self, event, image):
        # Event-Image Contrastive Learning
        with torch.no_grad():
            image_features = self.image_encoder(image)                                                  # N×D
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-9)

        event_features = self.event_encoder(event)
        event_features = event_features / (event_features.norm(dim=-1, keepdim=True) + 1e-9)

        logits_per_event = torch.exp(self.event_logit_scale) * event_features @ image_features.t()      # N×N
        logits_per_image = logits_per_event.t()                                                         # N×N

        ground_truth = torch.arange(len(logits_per_event)).long().to(logits_per_event.device)
        itc_loss = (
            F.cross_entropy(logits_per_event, ground_truth)
            + F.cross_entropy(logits_per_image, ground_truth)
        ) / 2

        return itc_loss
    
    def encode_event(self, event):
        with torch.no_grad():
            event_features = self.event_encoder(event)
            event_features = event_features / (event_features.norm(dim=-1, keepdim=True) + 1e-9)
        
        return event_features



