import torch
import os
from torch import nn
from PIL import Image
import numpy as np
from src.models.components.multimodal_transformer import (
    FLAVATransformerWithoutEmbeddings,
    TransformerEncoder,
)
from src.models.components.text_encoder import BertEncoder
from src.models.components.vision_encoder import ImageEncoder
from src.models.components.normalizations import Fp32LayerNorm
from src.losses import Pooler
from typing import Any, Callable, List, Tuple, Union
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoImageProcessor
import torchvision.transforms as transforms
from src.utils.utils import build_transformation


class MedMoE(nn.Module):
    def __init__(
        self,
        vision: DictConfig,
        text: DictConfig,
        lora: bool = False,
    ) -> None:

        super().__init__()
        self.text = text
        self.vision = vision
        self.tokenizer = AutoTokenizer.from_pretrained(text.tokenizer)
        self.image_processor = AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        self.image_encoder = ImageEncoder(vision)
        self.text_encoder = BertEncoder(text)
        self.text_encoder.output_tokens = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if vision.checkpoint_path and os.path.isfile(vision.checkpoint_path):
            state_dict = torch.load(vision.checkpoint_path, map_location=self.device)
            if 'medclip' in vision.checkpoint_path.lower():
                state_dict = {k.replace('vision_model.', 'model.'): v for k, v in state_dict.items() if 'vision_model' in k}
            self.image_encoder.to(self.device)
            self.image_encoder.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded vision model weights from {vision.checkpoint_path}")
        else:
            print(f"WARNING: vision model weights not loaded from {vision.checkpoint_path}")

        if text.checkpoint_path and os.path.isfile(text.checkpoint_path):       
            state_dict = torch.load(text.checkpoint_path, map_location=self.device)
            if 'medclip' in text.checkpoint_path.lower():
                text_state_dict = {k.replace('text_model.', ''): v for k, v in state_dict.items() if 'text_model' in k}
                self.text_encoder.to(self.device)
                self.text_encoder.load_state_dict(text_state_dict, strict=False)
                print(f"Successfully loaded text model weights from {text.checkpoint_path}")
            else:
                print(f"WARNING: text model weights not loaded from {text.checkpoint_path}")
                
                # not implemented for other weights
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"Successfully loaded model weights from {text.checkpoint_path}")

        self.image_encoder.to(self.device)
        self.text_encoder.to(self.device)

    def encode_image(self, images):
        self.image_encoder.to(self.device)
        img_feat_g, local_feats, router_logits = self.image_encoder(images, get_local=True)
        return img_feat_g, local_feats, router_logits
    
    def encode_text(self, texts):
        sentences = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",    
            padding='longest',            
            truncation=True,      
            max_length=self.text.max_length
        )
        self.text_encoder.to(self.device)
        text_emb_l, text_emb_g, sents = self.text_encoder(
            sentences['input_ids'].to(self.device), 
            sentences['attention_mask'].to(self.device), 
            sentences['token_type_ids'].to(self.device)
        ) 
        
        if self.text.projection: # not tested 
            text_emb_l = self.text_projection(text_emb_l) 
            text_emb_g = self.text_projection(text_emb_g)
            return text_emb_l, text_emb_g, sents
        
        if self.text.norm:
            text_emb_l = text_emb_l / torch.norm(
                text_emb_l, 2, dim=1, keepdim=True
            ).expand_as(text_emb_l)
            text_emb_g = text_emb_g / torch.norm(
                text_emb_g, 2, dim=1, keepdim=True
            ).expand_as(text_emb_g)
            
        return text_emb_l, text_emb_g, sents
        
    def forward(self, batch):

        images, text = batch['image'], batch['caption']
        
        text_emb_l, text_emb_g, sents = self.encode_text(text)  
        img_emb_g, img_emb_l, router_logits = self.encode_image(images)
        return img_emb_g, img_emb_l, text_emb_g, text_emb_l, sents, router_logits
    
        
if __name__ == "__main__":
    _ = MedMoE()