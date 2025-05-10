# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, OrderedDict, Union, Tuple, Dict, Callable, List 

import torch
from torch import nn, Tensor
from src.models.components.normalizations import Fp32LayerNorm
from src.utils.common import ModelOutput
from src.utils.distributed import BackpropType
import torch.nn.functional as F 

def assert_labels_are_present(
    labels: Optional[Tensor], category: str = "labels"
) -> None:
    assert (
        labels is not None
    ), f"Model is in training model but {category} are not passed"


@dataclass
class ITMLossOutput(ModelOutput):
    logits: Tensor
    loss: Tensor


@dataclass
class MaskedPredictionLossOutput(ModelOutput):
    logits: Tensor
    loss: Tensor



@dataclass
class GLORIALocalContrastiveLossOutput(OrderedDict):
    loss0: Tensor
    loss1: Tensor
    att_maps: List[Tensor]
    
@dataclass
class GLORIAGlobalContrastiveLossOutput:
    loss: Tensor
    
@dataclass
class FLAVAGlobalContrastiveLossOutput(OrderedDict):
    text_embedding: Tensor
    image_embedding: Tensor
    logit_scale: Tensor
    image_logits: Tensor
    text_logits: Tensor
    image_loss: Tensor
    text_loss: Tensor
    loss: Tensor


@dataclass
class FLAVAPretrainingLossesCollection(ModelOutput):
    mmm_text_loss: Optional[Tensor] = None
    mmm_image_loss: Optional[Tensor] = None
    mim_loss: Optional[Tensor] = None
    mlm_loss: Optional[Tensor] = None
    itm_loss: Optional[Tensor] = None
    global_contrastive_loss: Optional[Tensor] = None


@dataclass
class FLAVAPretrainingLossOutput(ModelOutput):
    losses: FLAVAPretrainingLossesCollection = field(
        default_factory=FLAVAPretrainingLossesCollection
    )
    mlm_output: Optional[MaskedPredictionLossOutput] = None
    mim_output: Optional[MaskedPredictionLossOutput] = None
    mmm_text_output: Optional[MaskedPredictionLossOutput] = None
    mmm_image_output: Optional[MaskedPredictionLossOutput] = None
    itm_output: Optional[ITMLossOutput] = None
    global_contrastive_output: Optional[FLAVAGlobalContrastiveLossOutput] = None
    image_sequence: Optional[Tensor] = None
    text_sequence: Optional[Tensor] = None
    image_masked_sequence: Optional[Tensor] = None
    text_masked_sequence: Optional[Tensor] = None
    multimodal_sequence: Optional[Tensor] = None
    multimodal_masked_sequence: Optional[Tensor] = None


# TODO(asg): Replace later with MLP classifier if checkpoint permits
class Pooler(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs: Any):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class TwoWayHead(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs: Any):
        super().__init__()

        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:
        return self.seq_relationship(pooled_output)


class ITMLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__()
        self.pooler = Pooler(hidden_size=hidden_size)
        self.cls = TwoWayHead(hidden_size=hidden_size)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor,
    ) -> ITMLossOutput:
        if self.training:
            assert_labels_are_present(labels, "itm labels")

        pooled_output = self.pooler(hidden_states)
        scores = self.cls(pooled_output)

        if labels is None:
            loss = pooled_output.sum() * 0
        else:
            loss = self.ce_loss(
                scores.view(-1, 2),
                labels.view(-1),
            )
        return ITMLossOutput(logits=scores, loss=loss)


class MaskedPredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        use_fp32_layer_norm: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn

        self.layer_norm: nn.LayerNorm
        if use_fp32_layer_norm:
            self.layer_norm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is
        # correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MaskedPredictionLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        ignore_index: int = -1,
        ignore_nan: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        self.cls = MaskedPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            transform_act_fn=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
        )
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_nan = ignore_nan

    def forward(
        self, hidden_states: Tensor, masked_labels: Optional[Tensor] = None
    ) -> MaskedPredictionLossOutput:
        if self.training:
            assert_labels_are_present(masked_labels, "masked labels")

        if masked_labels is not None:
            masked_tokens = masked_labels.ne(self.ignore_index)
            masked_labels = masked_labels[masked_tokens]
            sequence_output = hidden_states[masked_tokens, :]
        else:
            sequence_output = hidden_states

        prediction = self.cls(sequence_output)

        if masked_labels is None:
            masked_loss = prediction.sum() * 0
        else:
            masked_loss = self.ce_loss(
                prediction.view(-1, self.vocab_size),
                masked_labels.view(-1),
            )

        # When masked_labels are all ignore_index then masked_lm_loss is NaN,
        # so we replace NaN with 0.
        if torch.isnan(masked_loss) and self.ignore_nan:
            warnings.warn("NaN detected in masked_loss. Replacing it with 0.")
            masked_loss = torch.nan_to_num(masked_loss, nan=0.0)

        return MaskedPredictionLossOutput(
            logits=prediction,
            loss=masked_loss,
        )


class FLAVAGlobalContrastiveLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
        image_embedding_size: int = 768,
        text_embedding_size: int = 768,
        projection_size: int = 768,
        image_embedding_index: int = 0,
        text_embedding_index: int = 0,
    ):
        super().__init__()
        if logit_scale is None:
            logit_scale = math.log(1 / 0.07)

        # If already initialized, set to what was passed
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def forward(
        self,
        image_sequence: Tensor,
        text_sequence: Tensor,
        mask: Optional[Tensor] = None,
    ) -> FLAVAGlobalContrastiveLossOutput:

        text_embedding = nn.functional.normalize(text_sequence, dim=-1)
        image_embedding = nn.functional.normalize(
            image_sequence,
            dim=-1,
        )

        self.logit_scale.data.clamp_(0, 4.6052)

        output = contrastive_loss_with_temperature(
            embeddings_a=image_embedding,
            embeddings_b=text_embedding,
            logit_scale=self.logit_scale,
            mask=mask,
            # Always true for FLAVA global contrastive loss
            backprop_type=BackpropType.GLOBAL,
        )

        return FLAVAGlobalContrastiveLossOutput(
            loss=output.loss,
            image_logits=output.logits_a,
            text_logits=output.logits_b,
            image_loss=output.loss_a,
            text_loss=output.loss_b,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            logit_scale=self.logit_scale.data,
        )


class FLAVAPretrainingLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
        hidden_size: int = 768,
        text_vocab_size: int = 30522,
        image_vocab_size: int = 8192,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        ignore_index: int = -1,
        mlm_weight: float = 1.0,
        mim_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        mmm_image_loss_weight: float = 1.0,
        mmm_text_loss_weight: float = 1.0,
        itm_loss_weight: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()

        self.contrastive_loss = FLAVAGlobalContrastiveLoss(
            logit_scale=logit_scale,
            image_embedding_size=hidden_size,
            text_embedding_size=hidden_size,
            projection_size=hidden_size,
        )
        self.mlm_loss = MaskedPredictionLoss(
            hidden_size=hidden_size,
            vocab_size=text_vocab_size,
            transform_act_fn=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
            ignore_index=ignore_index,
        )
        self.mim_loss = MaskedPredictionLoss(
            hidden_size=hidden_size,
            vocab_size=image_vocab_size,
            transform_act_fn=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
            ignore_index=ignore_index,
        )
        # Create separate weights for MMM loss
        self.mmm_loss = nn.ModuleDict(
            {
                "mlm": MaskedPredictionLoss(
                    hidden_size=hidden_size,
                    vocab_size=text_vocab_size,
                    transform_act_fn=transform_act_fn,
                    layer_norm_eps=layer_norm_eps,
                    ignore_index=ignore_index,
                ),
                "mim": MaskedPredictionLoss(
                    hidden_size=hidden_size,
                    vocab_size=image_vocab_size,
                    transform_act_fn=transform_act_fn,
                    layer_norm_eps=layer_norm_eps,
                    ignore_index=ignore_index,
                ),
            }
        )
        self.itm_loss = ITMLoss(
            hidden_size=hidden_size,
            ignore_index=ignore_index,
        )

        self.mim_weight = mim_weight
        self.mlm_weight = mlm_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.mmm_image_loss_weight = mmm_image_loss_weight
        self.mmm_text_loss_weight = mmm_text_loss_weight
        self.itm_loss_weight = itm_loss_weight

    # TODO: Some refactoring is needed in this function to make it look better
    # TODO: Possibly refactor this into functional and class component
    # for better usability
    def forward(
        self,
        image_sequence: Optional[Tensor] = None,
        text_sequence: Optional[Tensor] = None,
        image_masked_sequence: Optional[Tensor] = None,
        text_masked_sequence: Optional[Tensor] = None,
        multimodal_sequence: Optional[Tensor] = None,
        multimodal_masked_sequence: Optional[Tensor] = None,
        itm_labels: Optional[Tensor] = None,
        mim_labels: Optional[Tensor] = None,
        mlm_labels: Optional[Tensor] = None,
        projected_image_embeddings: Optional[Tensor] = None,
        projected_text_embeddings: Optional[Tensor] = None,
    ) -> FLAVAPretrainingLossOutput:
        outputs = FLAVAPretrainingLossOutput()
        pos_mask = None

        # Check multimodal_masked_sequence to make sure this is unimodal case
        # This specific case can though be backpropagated directly as MIM is independent of
        # text, but that is a research question :)

        if (
            image_masked_sequence is not None
            and self.mim_weight > 0
            and multimodal_masked_sequence is None
        ):
            # Remove CLS token from image_masked_sequence

            start_index = -mim_labels.size(1) if mim_labels is not None else 1
            outputs.mim_output = self.mim_loss(
                image_masked_sequence[:, start_index:, :], mim_labels
            )
            outputs.mim_output.loss *= self.mim_weight
            outputs.losses.mim_loss = outputs.mim_output.loss

        # Check multimodal_masked_sequence to make sure this is unimodal case
        if (
            text_masked_sequence is not None
            and self.mlm_weight > 0
            and multimodal_masked_sequence is None
        ):
            start_index = -mlm_labels.size(1) if mlm_labels is not None else 1
            outputs.mlm_output = self.mlm_loss(
                text_masked_sequence[:, start_index:, :], mlm_labels
            )
            outputs.mlm_output.loss *= self.mlm_weight
            outputs.losses.mlm_loss = outputs.mlm_output.loss

        if multimodal_masked_sequence is not None and self.itm_loss_weight > 0:
            if itm_labels is not None:
                pos_pairs = itm_labels.ne(0)
                pos_mask = torch.where(
                    pos_pairs.any(), pos_pairs, pos_pairs.new([True])
                )
            else:
                pos_mask = torch.ones(
                    multimodal_masked_sequence.size(0),
                    device=multimodal_masked_sequence.device,
                ).bool()
            outputs.itm_output = self.itm_loss(multimodal_masked_sequence, itm_labels)
            outputs.itm_output.loss *= self.itm_loss_weight
            outputs.losses.itm_loss = outputs.itm_output.loss

            multimodal_masked_sequence = multimodal_masked_sequence[pos_mask]
            if mlm_labels is not None:
                mlm_labels = mlm_labels[pos_mask]
            if mim_labels is not None:
                mim_labels = mim_labels[pos_mask]

        if multimodal_masked_sequence is not None and self.mmm_text_loss_weight > 0:
            start_index = (
                -mlm_labels.size(1)
                if mlm_labels is not None
                else -(text_masked_sequence.size(1) - 1)
            )
            sequence_for_text = multimodal_masked_sequence[:, start_index:, :]
            outputs.mmm_text_output = self.mmm_loss.mlm(
                sequence_for_text,
                mlm_labels,
            )  # type: ignore
            outputs.mmm_text_output.loss *= self.mmm_text_loss_weight
            outputs.losses.mmm_text_loss = outputs.mmm_text_output.loss

        if multimodal_masked_sequence is not None and self.mmm_image_loss_weight > 0:
            # Starts from 2 because of 2 CLS, one for multimodal encoder and one
            # that comes from image encoder.
            total_indices = (
                mim_labels.size(1)
                if mlm_labels is not None
                else (image_masked_sequence.size(1) - 1)
            )
            sequence_for_image = multimodal_masked_sequence[:, 2 : 2 + total_indices, :]
            outputs.mmm_image_output = self.mmm_loss.mim(
                sequence_for_image,
                mim_labels,
            )  # type: ignore
            outputs.mmm_image_output.loss *= self.mmm_image_loss_weight
            outputs.losses.mmm_image_loss = outputs.mmm_image_output.loss

        if (
            projected_image_embeddings is not None
            and projected_text_embeddings is not None
            and self.contrastive_loss_weight > 0
        ):
            outputs.global_contrastive_output = self.contrastive_loss(
                projected_image_embeddings,
                projected_text_embeddings,
                pos_mask,
            )
            outputs.global_contrastive_output.loss *= self.contrastive_loss_weight
            outputs.losses.global_contrastive_loss = (
                outputs.global_contrastive_output.loss
            )

        return outputs

@dataclass
class ContrastiveLossOutput(OrderedDict):
    loss: Tensor
    logits_a: Tensor
    logits_b: Tensor
    loss_a: Tensor
    loss_b: Tensor


def _gather_embeddings_and_labels(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    backprop_type: BackpropType = BackpropType.GLOBAL,
) -> Tuple[Tensor, Tensor, Tensor]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        return embeddings_a, embeddings_b, labels

    embeddings_a_all_gpus = gather_tensor(embeddings_a, backprop_type)
    embeddings_b_all_gpus = gather_tensor(embeddings_b, backprop_type)
    # embeddings_a has shape [local_batch_size, embedding_dim]
    local_batch_size = embeddings_a.size(0)
    labels = local_batch_size * torch.distributed.get_rank() + torch.arange(
        local_batch_size, device=embeddings_a.device
    )

    return (
        torch.cat(embeddings_a_all_gpus),
        torch.cat(embeddings_b_all_gpus),
        labels,
    )


def contrastive_loss_with_temperature(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    logit_scale: nn.Parameter,
    mask: Optional[Tensor] = None,
    backprop_type: BackpropType = BackpropType.GLOBAL,
    cross_entropy_kwargs: Optional[Dict[str, Any]] = None,
) -> ContrastiveLossOutput:
    """Functional component for the ContrastiveLossWithTemperature. Please
    check the class for more details

    Args:
        embeddings_a (Tensor): Tensor containing features from the first input or modality.
            (In the CLIP model, these are the outputs of the image encoder.)
        embeddings_b (Tensor): Tensor containing features from the second input or modality.
            (In the CLIP model, these are the outputs of the text encoder.)
        logit_scale (nn.Parameter): Parameter with value of log of the learned temperature
        mask (Optional[Tensor], optional): If certain elements of the inputs shouldn't
            be considered in the loss calculation use this option to pass a boolean
            mask. Size is (BatchSize,). Defaults to None.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL
        cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)

    Returns:
        ContrastiveLossOutput: instance of ContrastiveLossOutput with all of the
            relevant fields.
    """

    # this temperature implementation follows CLIP Figure 3
    temperature = torch.exp(logit_scale)

    (
        embeddings_a_all_gpus,
        embeddings_b_all_gpus,
        labels,
    ) = _gather_embeddings_and_labels(embeddings_a, embeddings_b, backprop_type)

    # logits_per_image has shape [local_batch_size, global_batch_size]
    logits_per_input_a = (
        torch.matmul(embeddings_a, embeddings_b_all_gpus.transpose(0, 1)) * temperature
    )
    logits_per_input_b = (
        torch.matmul(embeddings_b, embeddings_a_all_gpus.transpose(0, 1)) * temperature
    )

    if mask is not None:
        logits_per_input_a = logits_per_input_a[mask]
        logits_per_input_b = logits_per_input_b[mask]
        labels = labels[mask]

    if cross_entropy_kwargs is None:
        cross_entropy_kwargs = {}

    loss_a = F.cross_entropy(logits_per_input_a, labels, **cross_entropy_kwargs)
    loss_b = F.cross_entropy(logits_per_input_b, labels, **cross_entropy_kwargs)
    loss = (loss_a + loss_b) / 2

    return ContrastiveLossOutput(
        loss=loss,
        logits_a=logits_per_input_a,
        logits_b=logits_per_input_b,
        loss_a=loss_a,
        loss_b=loss_b,
    )


DEFAULT_LOGIT_SCALE = math.log(1 / 0.07)

'''
class ContrastiveLossWithTemperature(nn.Module):
    """Contrastive loss with a temperature parameter, as used in CLIP and FLAVA.
    CLIP: https://arxiv.org/pdf/2103.00020.pdf
    FLAVA: https://arxiv.org/pdf/2112.04482.pdf


    A contrastive loss over pairs of input embeddings a and b. For each input_a
    embedding, we compute a weighted cosine similarity with all input_b embeddings,
    then calculate the cross entropy loss against the true (input_a, input_b) pairing.
    Each input_b embedding is evaluated against all input_a embeddings similarly.
    The batch's loss is the average cross entropy over all input_a and input_b embeddings
    in the batch.

    Temperature is a learned parameter clamped to ``[1, 100]`` and
    initialized to 1 / 0.07 as in the CLIP paper.


    Args:
        logit_scale (Union[float, nn.Module]): Log of the learnable temperature parameter value
            A nn.Parameter instantiation can also be passed directly in case parent class
            is handling the initialization.
            Defaults to ``ln(1/0.07)``, as in the CLIP paper.
        logit_scale_min (Optional[float]): Log of the minimum temperature value.
            If ``None``, then temperature will not be clamped to a minimum value.
            Defaults to ``ln(1)``, as in the CLIP paper.
        logit_scale_max (Optional[float]): Log of the maximum temperature value.
            If ``None``, then temperature will not be clamped to a maximum value.
            Defaults to ``ln(100)``, as in the CLIP paper.

    Inputs: embeddings_a (Tensor): Tensor containing features from the first input or modality.
                (In the CLIP model, these are the outputs of the image encoder.)
            embeddings_b (Tensor): Tensor containing features from the second input or modality.
                (In the CLIP model, these are the outputs of the text encoder.)
            backprop_type (BackpropType): whether to backpropagate gradients to all
                workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
                Default: BackpropType.GLOBAL
            cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)
            mask (Optional[Tensor], optional): If certain elements of the inputs shouldn't
                be considered in the loss calculation use this option to pass a boolean
                mask. Size is (BatchSize,). Defaults to None.
    """

    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = DEFAULT_LOGIT_SCALE,
        logit_scale_min: Optional[float] = math.log(1),
        logit_scale_max: Optional[float] = math.log(100),
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        if not logit_scale_min and not logit_scale_max:
            raise ValueError(
                "Only one of `logit_scale_min` and `logit_scale_max` can be None."
            )
        self.logit_scale_min = logit_scale_min
        self.logit_scale_max = logit_scale_max

        # If already initialized, set to what was passed
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        backprop_type: BackpropType = BackpropType.GLOBAL,
        cross_entropy_kwargs: Optional[Dict[str, Any]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        self.logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)
        return contrastive_loss_with_temperature(
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
            logit_scale=self.logit_scale,
            backprop_type=backprop_type,
            cross_entropy_kwargs=cross_entropy_kwargs,
            mask=mask,
        ).loss
'''

#################################
# Following from GLOria Implementation
# https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py
#################################

from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)



class ZEROGlobalContrastiveLoss(nn.Module):
    
    def __init__(
        self,
    ):
        super(ZEROGlobalContrastiveLoss, self).__init__()

    def forward(
        self,
        cnn_code: Tensor,
        rnn_code: Tensor,
        temp3: float = 10.0,
        idx: int = None, 
        probs: Tensor = None,
    ) -> Tensor:
        return torch.tensor(0.0)
        
class GLORIAGlobalContrastiveLoss(nn.Module):
    
    def __init__(
        self,
    ):
        super(GLORIAGlobalContrastiveLoss, self).__init__()
        self.eps = 1e-8
        self.temp3 = 10.0

    def forward(
        self,
        cnn_code: Tensor,
        rnn_code: Tensor,
        temp3: float = 10.0,
        idx: int = None, 
        probs: Tensor = None,
    ) -> Tensor:

        batch_size = cnn_code.shape[0]
        labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        cnn_code_norm = torch.norm(cnn_code, 2, dim=-1, keepdim=True) # [1, 128, 768] [1, 128, 768]
        rnn_code_norm = torch.norm(rnn_code, 2, dim=-1, keepdim=True) # [1, 128, 768]
        
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=self.eps) * temp3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        scores1 = scores0.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        return loss0 + loss1

def softXEnt(target, logits):
    """
    From the pytorch discussion Forum:
    https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
    """
    logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
    loss = -(target * logprobs).sum() / logits.shape[0]
    return loss

def softXEntPenalty(target, logits,penalty):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
        loss = -(target * logprobs * penalty).sum() / logits.shape[0]
        return loss
    
class SoftGLORIAGlobalContrastiveLoss(nn.Module):
    '''
    Adapted from SAT implementation 
    https://github.com/liubo105/SAT/blob/main/modules/gloria_loss.py
    '''
    
    def __init__(
        self,
    ):
        super(SoftGLORIAGlobalContrastiveLoss, self).__init__()
        self.eps = 1e-8

    def forward(
        self,
        cnn_code: Tensor,
        rnn_code: Tensor,
        temp3: float = 10.0,
        idx: int = None, 
        probs: Tensor = None,
    ) -> GLORIALocalContrastiveLossOutput:

        batch_size = cnn_code.shape[0]
        labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        cnn_code_norm = torch.norm(cnn_code, 2, dim=-1, keepdim=True) # [1, 128, 768] [1, 128, 768]
        rnn_code_norm = torch.norm(rnn_code, 2, dim=-1, keepdim=True) # [1, 128, 768]
        
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=self.eps) * temp3

        scores0 = scores0.squeeze()
        scores1 = scores0.transpose(0, 1)
        
        loss0 = 0
        loss1 = 0
        scores = idx
        threshold1, threshold2 = probs

        for layer,i in enumerate(scores):
            pos  = (i>threshold1).nonzero().squeeze(-1)
            neg = (i<=threshold2).nonzero().squeeze(-1)
            neg_i2t = scores0[layer][neg]
            neg_t2i = scores1[layer][neg]
            loss_i2t = 0
            loss_t2i = 0

            for j in pos:
                pos_i2t = scores0[layer][j].unsqueeze(-1)
                pos_t2i = scores1[layer][j].unsqueeze(-1)
                new_i2t = torch.cat([pos_i2t,neg_i2t])
                new_t2i = torch.cat([pos_t2i,neg_t2i])
                targets = torch.zeros_like(new_i2t,dtype=torch.long)
                targets[0] = 1
                loss_i2t += softXEnt(targets,new_i2t)
                loss_t2i += softXEnt(targets,new_t2i)
    
            loss_i2t /= len(pos)
            loss_t2i /= len(pos)
            loss0 += loss_i2t
            loss1 += loss_t2i

        loss0 /= batch_size
        loss1 /= batch_size
        
        return loss0 + loss1
    
class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, 
        imgs: Tensor,
        caps: Tensor,
        temp3: float = 10.0, # placeholder not used 
        idx: int = None, 
        probs: Tensor = None,
    ): 
        caps = nn.functional.normalize(caps, dim=-1)
        imgs = nn.functional.normalize(imgs, dim=-1)
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()

        # Reducing the score on diagonal so there are not selected as hard negative
        scores = scores - 2 * torch.diag(scores.diag())

        sorted_cap, _ = torch.sort(scores, 0, descending=True)
        sorted_img, _ = torch.sort(scores, 1, descending=True)

        # Selecting the nmax hardest negative examples
        max_c = sorted_cap[: self.nmax, :]
        max_i = sorted_img[:, : self.nmax]

        # Margin based loss with hard negative instead of random negative
        neg_cap = torch.sum(
            torch.clamp(
                max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0
            )
        )
        neg_img = torch.sum(
            torch.clamp(
                max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0
            )
        )

        loss = neg_cap + neg_img

        return loss
    
class ZEROLocalContrastiveLoss(nn.Module):
    
    def __init__(
        self,
    ):
        super(ZEROLocalContrastiveLoss, self).__init__()

    def forward(
        self,
        img_features: Tensor,
        words_emb: Tensor,
        cap_lens: List[float],
        temp1: float = 4.0,
        temp2: float = 5.0,
        temp3: float = 10.0,
        agg: str = "sum",
        idx: int = None, 
        probs: Tensor = None
    ) -> GLORIALocalContrastiveLossOutput:
        return GLORIALocalContrastiveLossOutput(
            loss1=torch.tensor(0.0),
            loss0=torch.tensor(0.0),
            att_maps=[]
        )

class GLORIALocalContrastiveLoss(nn.Module):
    
    def __init__(
        self,
    ):
        super(GLORIALocalContrastiveLoss, self).__init__()
        
    def forward(
        self,
        img_features: Tensor,
        words_emb: Tensor,
        cap_lens: List[float],
        temp1: float = 4.0,
        temp2: float = 5.0,
        temp3: float = 10.0,
        agg: str = "sum",
        idx: int = None, 
        probs: Tensor = None,
    ) -> GLORIALocalContrastiveLossOutput:

        batch_size = img_features.shape[0]

        att_maps = []
        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):

            # Get the i-th text description
            words_num = cap_lens[i]  # 25
            # TODO: remove [SEP]
            # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_features  # [48, 768, 19, 19]

            weiContext, attn = attention_fn(
                word, context, temp1
            )  # [48, 768, 25], [48, 25, 19, 19]

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 19, 19]
            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        return GLORIALocalContrastiveLossOutput(
            loss1=loss1,
            loss0=loss0,
            att_maps=att_maps,
        )

# class GLORIALocalContrastiveLoss(nn.Module):
#     def __init__(self):
#         super(GLORIALocalContrastiveLoss, self).__init__()

#     def forward(
#         self,
#         img_features: List[Tensor],  # list of [B, D, H, W] per FPN level
#         words_emb: Tensor,           # [B, D, max_words]
#         cap_lens: List[int],
#         temp1: float = 4.0,
#         temp2: float = 5.0,
#         temp3: float = 10.0,
#         agg: str = "sum",
#         idx: int = None,
#         probs: Tensor = None,
#     ) -> GLORIALocalContrastiveLossOutput:

#         batch_size = words_emb.shape[0]
#         num_levels = len(img_features)
#         att_maps = []
#         similarities = []

#         for i in range(batch_size):
#             words_num = cap_lens[i]
#             word = words_emb[i, :, :words_num].unsqueeze(0).repeat(batch_size, 1, 1)  # [B, D, W]
#             word_t = word.transpose(1, 2).contiguous()  # [B, W, D]

#             # Step 1: Compute similarity per level (used for soft weighting)
#             level_sims = []
#             per_level_contexts = []

#             for l in range(num_levels):
#                 context = img_features[l]  # [B, D, H_l, W_l]
#                 weiContext, attn = attention_fn(word, context, temp1)  # [B, D, W], [B, W, H_l, W_l]

#                 per_level_contexts.append(weiContext)  # store for final agg
#                 sim = cosine_similarity(word_t.view(-1, word_t.shape[-1]),
#                                         weiContext.transpose(1, 2).contiguous().view(-1, weiContext.shape[1]))
#                 sim = sim.view(batch_size, words_num)
#                 sim_score = sim.mean(dim=1, keepdim=True)  # [B, 1]
#                 level_sims.append(sim_score)

#             level_sims_stack = torch.stack(level_sims, dim=-1)  # [B, 1, L]
#             level_weights = torch.softmax(level_sims_stack, dim=-1)  # soft assignment of levels per image

#             # Step 2: Soft aggregation of attention-weighted contexts
#             aggregated_context = 0
#             for l in range(num_levels):
#                 weiContext = per_level_contexts[l]  # [B, D, W]
#                 weiContext_t = weiContext.transpose(1, 2).contiguous()  # [B, W, D]
#                 weight_l = level_weights[:, :, l].unsqueeze(-1)  # [B, 1, 1]
#                 aggregated_context += weiContext_t * weight_l  # broadcast over words

#             word_t_flat = word_t.view(batch_size * words_num, -1)
#             context_flat = aggregated_context.view(batch_size * words_num, -1)

#             row_sim = cosine_similarity(word_t_flat, context_flat)
#             row_sim = row_sim.view(batch_size, words_num)

#             row_sim.mul_(temp2).exp_()
#             if agg == "sum":
#                 row_sim = row_sim.sum(dim=1, keepdim=True)  # [B, 1]
#             else:
#                 row_sim = row_sim.mean(dim=1, keepdim=True)  # [B, 1]
#             row_sim = torch.log(row_sim)

#             similarities.append(row_sim)

#         similarities = torch.cat(similarities, dim=1)  # [B, B]
#         similarities *= temp3
#         similarities_t = similarities.transpose(0, 1)

#         labels = torch.arange(batch_size).to(similarities.device)
#         loss0 = F.cross_entropy(similarities, labels)
#         loss1 = F.cross_entropy(similarities_t, labels)

#         return GLORIALocalContrastiveLossOutput(
#             loss1=loss1,
#             loss0=loss0,
#             att_maps=att_maps  # Optional: currently not retained
#         )


class SoftGLORIALocalContrastiveLoss(nn.Module):
    '''
    Adapted from SAT implementation 
    https://github.com/liubo105/SAT/blob/main/modules/gloria_loss.py
    '''
    
    def __init__(
        self,
    ):
        super(SoftGLORIALocalContrastiveLoss, self).__init__()
        
    def forward(
        self,
        img_features: Tensor,
        words_emb: Tensor,
        cap_lens: List[float],
        temp1: float = 4.0,
        temp2: float = 5.0,
        temp3: float = 10.0,
        agg: str = "sum",
        idx: int = None, 
        probs: Tensor = None
    ) -> GLORIALocalContrastiveLossOutput:

        batch_size = img_features.shape[0]

        att_maps = []
        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):

            # Get the i-th text description
            words_num = cap_lens[i]  # 25
            # TODO: remove [SEP]
            # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_features  # [48, 768, 19, 19]

            weiContext, attn = attention_fn(
                word, context, temp1
            )  # [48, 768, 25], [48, 25, 19, 19]

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 19, 19]
            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]

        loss0 = 0
        loss1 = 0
        scores = idx
        threshold1, threshold2 = probs

        for layer,i in enumerate(scores):
            pos  = (i>threshold1).nonzero().squeeze(-1)
            neg = (i<=threshold2).nonzero().squeeze(-1)
            neg_i2t = similarities[layer][neg]
            neg_t2i = similarities1[layer][neg]
            loss_i2t = 0
            loss_t2i = 0

            for j in pos:
                pos_i2t = similarities[layer][j].unsqueeze(-1)
                pos_t2i = similarities1[layer][j].unsqueeze(-1)
                new_i2t = torch.cat([pos_i2t,neg_i2t])
                new_t2i = torch.cat([pos_t2i,neg_t2i])
                targets = torch.zeros_like(new_i2t,dtype=torch.long)
                targets[0] = 1
                loss_i2t += softXEnt(targets,new_i2t)
                loss_t2i += softXEnt(targets,new_t2i)
    
            loss_i2t /= len(pos)
            loss_t2i /= len(pos)
            loss0 += loss_i2t
            loss1 += loss_t2i

        loss0 /= batch_size
        loss1 /= batch_size
        
        return GLORIALocalContrastiveLossOutput(
            loss1=loss1,
            loss0=loss0,
            att_maps=att_maps,
        )
