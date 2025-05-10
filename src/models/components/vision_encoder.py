#from numpy.lib.function_base import extract
from numpy import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model



class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg

        self.output_dim = cfg.embed_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_function = getattr(cnn_backbones, cfg.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=cfg.pretrained,
            lora=cfg.lora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_moe=cfg.use_moe,
        )

        # config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     target_modules=["query", "value"],
        #     lora_dropout=0.1,
        # )
        # self.model = get_peft_model(self.base_model, config)
        # del self.base_model
        
        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if cfg.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        
        if "swin" in self.cfg.model_name:
            global_ft, local_ft, router_logits = self.model(x)
            return global_ft, local_ft, router_logits

        elif "resnet" or "resnext" in self.cfg.model_name:
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)
            # print(global_ft.shape, local_ft.shape)
            # exit()
        elif "densenet" in self.cfg.model_name:
            global_ft, local_ft = self.dense_forward(x, extract_features=True)

        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features=None):

        global_emb = self.global_embedder(global_features)
        if local_features is not None:      
            local_emb = self.local_embedder(local_features)
        else:
            local_emb = None

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    # def resnet_forward(self, x, extract_features=False):

    #     # --> fixed-size input: batch x 3 x 299 x 299
    #     x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

    #     x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
    #     x = self.model.bn1(x)
    #     x = self.model.relu(x)
    #     x = self.model.maxpool(x)

    #     f1 = self.model.layer1(x)  # (batch_size, 64, 75, 75)
    #     f2 = self.model.layer2(f1)  # (batch_size, 128, 38, 38)
    #     f3 = self.model.layer3(f2)  # (batch_size, 256, 19, 19)
    #     # local_features = x
    #     f4 = self.model.layer4(f3)  # (batch_size, 512, 10, 10)

    #     x = self.pool(f4)
    #     x = x.view(x.size(0), -1)

    #     return x, [f1, f2, f3, f4]
 
    def densenet_forward(self, x, extract_features=False):
        pass
    

    # def project_local(self, x, idx):
    #     return self.local_embedder_list[idx](x)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred