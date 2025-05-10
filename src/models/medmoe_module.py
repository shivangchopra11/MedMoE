from typing import Any, Dict, Tuple

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
import src.losses as loss
from omegaconf import DictConfig
import hydra
from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.nn.functional as F
from transformers import BertModel

class MedMoELitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int=6,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param model: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = model


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_c_acc = Accuracy(task="multiclass", num_classes=6)
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.model`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"Please implement the get_loss method for {self.__class__.__name__}")

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        raise NotImplementedError(f"Please implement the get_loss method for {self.__class__.__name__}")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss_dict = self.model_step(batch)
        self.val_loss(loss_dict["loss"])
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(val_loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss_dict = self.model_step(batch)
        self.test_loss(loss_dict["loss"])
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    
class MedMoEPretrainingLightningModule(MedMoELitModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 5,
    ):
        super().__init__(model, optimizer, scheduler, compile, num_classes)
        self.loss_cfg = loss 
        self.local_loss = loss.local_loss
        self.global_loss = loss.global_loss
        self.local_loss_weight = loss.get("local_loss_weight", 0.4) 
        self.global_loss_weight = loss.get("global_loss_weight", 0.4) 
        self.classifier_loss_weight = loss.get("classifier_loss_weight", 0.2)
        self.train_loss = MeanMetric()
        self.train_l_loss = MeanMetric()
        self.train_g_loss = MeanMetric()
        self.train_c_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        self.model.image_encoder.train()
        # self.model.text_encoder.train()
        # for name, param in self.model.named_parameters():
        #     if "image_encoder" in name:
        #         if 'lora' in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        #     else:
        #         param.requires_grad = False
        
        if self.loss_cfg.soft_label:
            self.tool_bert = BertModel.from_pretrained(
                self.model.text.bert_type
            )

    def _calc_global_loss(self, img_emb_g, text_emb_g, idx=None, probs=None):
        output = self.global_loss(
            img_emb_g, text_emb_g, 
            temp3 = self.loss_cfg.get("temp3", 4.0),
            idx = idx, probs = probs
        )
        return output

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents, idx=None, probs=None):
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        output = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1 = self.loss_cfg.get("temp1", 4.0),
            temp2 = self.loss_cfg.get("temp2", 5.0),
            temp3 = self.loss_cfg.get("temp3", 10.0),
            idx = idx, probs = probs
        )
        return output.loss0 + output.loss1

    def _calc_classifier_loss(self, router_logits, labels):
        loss = F.cross_entropy(router_logits, labels)
        return loss
    
    def _calc_classifier_acc(self, router_logits, labels):
        acc = (torch.argmax(router_logits, dim=1) == labels).float().mean()
        return acc
    
    def token_pooling(self, model_output, attention_mask=None):
        return model_output[:,0]
        '''
        # max pooling 
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]
        
        # sum pooling
        token_embeddings = model_output  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1)
        '''
    
    def get_text_soft_target(self, raw_txt, topK, threshold):
        '''
        adapted from https://github.com/liubo105/SAT/blob/main/models/gloria_model.py
        '''
        self.tool_bert.eval()
        with torch.no_grad():
            sentences = self.model.text_encoder.tokenizer.batch_encode_plus(
                raw_txt,
                return_tensors="pt",    
                padding='longest',            
                truncation=True,      
                max_length=self.model.text.max_length
            )
            sentences = sentences.to(self.device)
            f_txt = self.tool_bert(**sentences)

            # using mean features  
            f_txt = self.token_pooling(f_txt[0])
            f_txt = F.normalize(f_txt, p=2, dim=1)
            batch_size, d_k = f_txt.shape
            scores = torch.matmul(f_txt,f_txt.transpose(-2,-1)) 
            # val, idx = torch.topk(scores,topK,dim=-1)
            # filter = torch.masked_fill(val, val<threshold, 0)
            # return (idx, filter)
            return scores, threshold
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        img_emb_g, img_emb_l, text_emb_g, text_emb_l, sents, router_logits = self.forward(batch)
        
        # get soft labels if using SoftGloria losses 
        if self.loss_cfg.soft_label:
            idx, filter = self.get_text_soft_target(
                batch['caption'], self.loss_cfg.topk, (self.loss_cfg.threshold0, self.loss_cfg.threshold1)
            )
            l_loss = self._calc_local_loss(img_emb_l, text_emb_l, sents, idx, filter)
            g_loss = self._calc_global_loss(img_emb_g, text_emb_g, idx, filter)
            
        else:
            l_loss = self._calc_local_loss(img_emb_l, text_emb_l, sents)
            g_loss = self._calc_global_loss(img_emb_g, text_emb_g)

        print(torch.argmax(router_logits, dim=1), batch['label'])
        print(router_logits)

        classifier_loss = self._calc_classifier_loss(router_logits, batch['label'])
        classifier_acc = self._calc_classifier_acc(router_logits, batch['label'])

        loss = self.local_loss_weight * l_loss + self.global_loss_weight * g_loss + self.classifier_loss_weight * classifier_loss

        return {
            "loss": loss,
            "l_loss": l_loss,
            "g_loss": g_loss,
            "classifier_loss": classifier_loss,
            "classifier_acc": classifier_acc
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_dict = self.model_step(batch) 
        self.train_loss(loss_dict["loss"])
        self.train_l_loss(loss_dict["l_loss"])
        self.train_g_loss(loss_dict["g_loss"])
        self.train_c_loss(loss_dict["classifier_loss"])
        # self.train_c_acc(loss_dict["classifier_acc"])
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/l_loss", self.train_l_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/g_loss", self.train_g_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/c_loss", self.train_c_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/c_acc", loss_dict["classifier_acc"], on_step=True, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

