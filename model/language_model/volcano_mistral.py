from typing import List, Optional, Tuple, Union
from xml.dom.expatbuilder import parseString
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                        LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPImageProcessor
# from .modeling_llama import LlamaConfig, LlamaModel, LlamaForCausalLM
# from .modeling_mistral import MistralForCausalLM, MistralConfig, MistralModel
from transformers import MistralForCausalLM, MistralConfig, MistralModel,MistralPreTrainedModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from ..vision_encoder.builder import build_vision_encoder
# `````````from ..vision_generator.builder import build_vision_generator
from ..front_projector.builder import build_front_projector
# from ..behind_projector.builder import build_behind_projector
from .volcano_base import VolCanoMetaForCausalLM
from transformers import GenerationMixin

from PIL import Image
from constants import *
import random
from utils.util import rank_0_print
import torch.nn.functional as F
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.eval_util import extract_box_str
from locals.datasets.utils.box_utils import *


############################ quiet star ##################################
import inspect
import math
import copy
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from termcolor import colored
from tqdm import tqdm
import random
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from torch import nn
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from ...activations import ACT2FN
# from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
# from ...modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

############################ quiet star ##################################
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def resize_image_to_square(image):
    width, height = image.size
    max_side = max(width, height)
    image = image.resize((max_side, max_side))
    return image

def split_tensor_func(tensor, dim=0):
    num_splits = tensor.shape[dim]
    return [f.squeeze(dim) for f in tensor.split([1]*num_splits, dim=dim)]

@dataclass
class VolCanoCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    regression_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None

IGNORE_INDEX = -100
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
 
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        return super().forward(x)


class VolCanoMistralConfig(MistralConfig):
    model_type = "VolCanoMistral"
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,  # 注意这里初始化为None，以确保处理兼容性
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        # 新的参数
        max_thoughts=16,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        **kwargs,
    ):
        self.max_thoughts = max_thoughts
        self.merged_talk_heads = merged_talk_heads
        self.merged_lm_and_talk_heads = merged_lm_and_talk_heads
        self.merged_lm_and_think_heads = merged_lm_and_think_heads
        self.use_concat_talk_head = use_concat_talk_head
        self.use_shallow_think = use_shallow_think
        self.use_shallow_talk = use_shallow_talk
        self.use_complex_think_head = use_complex_think_head
        self.use_complex_talk_head = use_complex_talk_head
        self.use_weighted_talk_head = use_weighted_talk_head

        # 这里添加 num_key_value_heads 兼容性处理
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # 继承父类的配置
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,  # 确保传递处理过的参数
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            **kwargs,
        )    


class VolCanoMistralModel(MistralModel):
    config_class = VolCanoMistralConfig

    def __init__(self, config: MistralConfig):
        super(VolCanoMistralModel, self).__init__(config)
        

# class VolCanoMistralForCausalLM(MistralForCausalLM, VolCanoMetaForCausalLM):
#     config_class = VolCanoMistralConfig

#     def __init__(self, config):
#         super(MistralForCausalLM, self).__init__(config)
        
#         self.model = VolCanoMistralModel(config)

#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         self.init_vision_model()

#         if hasattr(self, 'vision_generator'):

#             zero_img_feature = torch.zeros((1, IMG_TOKEN_NUM, config.hidden_size))
#             self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

#         # Initialize weights and apply final processing
#         self.post_init()
#         self.output_img_id = -100
#         self.regression_weight = 1.0
#         self.no_bind = False
    
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         captions=[None],
#         output_image_feature=None,
#         output_images = None,
#         output_cond_images = None,
#         output_cond_img_mask = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         input_images: Optional[torch.FloatTensor] = None,
#         image_label_masks: Optional[torch.Tensor] = None,
#         return_dict: Optional[bool] = None,
#         item_id: Optional[bool] = None,
#         box: Optional[torch.Tensor] = None
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         # return torch.zeros(1).to(self.device).to(self.dtype)
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, visual_labels, visual_label_masks = self.prepare_inputs_labels_for_multimodal(
#             input_ids, position_ids ,attention_mask, past_key_values, labels = labels, images = input_images, image_label_masks=image_label_masks, inputs_embeds=inputs_embeds, box=box)
#         if inputs_embeds is not None:
#             inputs_embeds = inputs_embeds.to(self.dtype)
#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids = position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )
        
#         hidden_states = outputs[0]
#         logits = getattr(self.lm_head,'modules_to_save.default',self.lm_head)(hidden_states)
#         loss = None
#         text_loss = None
#         # compute text loss
#         if labels is not None:
#             # Shift so that tokens < n predict n

#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model/pipeline parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             text_loss = loss_fct(shift_logits, shift_labels)
#             # rank_0_print(text_loss)
#             # rank_0_print(labels)
            
#         # regression loss
#         visual_labels = None
#         if visual_labels is not None and text_loss is not None:
#             if visual_label_masks.sum() == 0:
#                 # # no target images
#                 # if text_loss is not None:
#                 #     regression_loss = torch.zeros_like(text_loss)
#                 # else:
#                 #     regression_loss = None
#                 last_hidden_state = hidden_states
#                 target_img_hidden_states = torch.masked_select(last_hidden_state[..., :-1, :], (visual_label_masks[:, 1:]>0).unsqueeze(-1)).reshape(-1, hidden_states.shape[-1])
#                 predict_image_feat = self.behind_projector(target_img_hidden_states)
#                 target_visual_labels = torch.masked_select(visual_labels, (visual_label_masks>0).unsqueeze(-1)).reshape(-1, visual_labels.shape[-1])
#                 regression_loss = F.mse_loss(predict_image_feat, target_visual_labels, reduction='none').sum()
#                 loss = self.regression_weight*regression_loss + (2 - self.regression_weight) * text_loss
#             else:
#                 last_hidden_state = hidden_states
#                 target_img_hidden_states = torch.masked_select(last_hidden_state[..., :-1, :], (visual_label_masks[:, 1:]>0).unsqueeze(-1)).reshape(-1, hidden_states.shape[-1])
#                 predict_image_feat = self.behind_projector(target_img_hidden_states)
#                 target_visual_labels = torch.masked_select(visual_labels, (visual_label_masks>0).unsqueeze(-1)).reshape(-1, visual_labels.shape[-1])
#                 regression_loss = F.mse_loss(predict_image_feat, target_visual_labels)

#                 if self.diffusion_loss:
#                     num_output_images = output_images.shape[0]
#                     predict_image_feat = predict_image_feat.reshape(-1, self.n_query, predict_image_feat.shape[-1])
#                     assert num_output_images == (predict_image_feat.shape[0]), 'the output images must match the images in sequences'
#                     random_probs = torch.rand(num_output_images).to(output_images.device).unsqueeze(-1).unsqueeze(-1)
#                     zero_image_feature = self.encode_img(torch.zeros(1, 3, self.vision_encoder.image_size, self.vision_encoder.image_size).to(device=output_images.device, dtype=output_images.dtype))[2]
#                     if USE_CFG:
#                         diffusion_input_feature = torch.where(random_probs < 0.1, zero_image_feature, predict_image_feat)
#                     else:
#                         diffusion_input_feature = predict_image_feat

#                     cond_img_mask = None
#                     output_cond_images = None

#                     if output_image_feature is None:
#                         image_loss = self.compute_image_loss(diffusion_input_feature, output_images, output_cond_image=output_cond_images, cond_img_mask=cond_img_mask)
#                     else:
#                         image_loss = self.compute_image_loss(diffusion_input_feature, None, output_cond_image=output_cond_images, cond_img_mask=cond_img_mask, output_image_feature=output_image_feature)

#                     loss = regression_loss + text_loss + image_loss
#                 else:
#                     # print("regression loss:{:.5f}".format(regression_loss.item()) + ' text loss:{:.5f}'.format(text_loss.item()))
#                     loss = self.regression_weight*regression_loss + (2 - self.regression_weight) * text_loss
#         else:
#             loss = text_loss
#             if text_loss is not None:
#                 regression_loss = torch.zeros_like(text_loss)
#             else:
#                 regression_loss = None


#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return VolCanoCausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             regression_loss=regression_loss,
#             text_loss=text_loss
#         )

#     def compute_snr(self,timesteps):
#         """
#         Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
#         """
#         alphas_cumprod = self.noise_scheduler.alphas_cumprod
#         sqrt_alphas_cumprod = alphas_cumprod**0.5
#         sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

#         # Expand the tensors.
#         # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
#         sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
#         while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
#             sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
#         alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

#         sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
#         while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
#             sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
#         sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

#         # Compute SNR.
#         snr = (alpha / sigma) ** 2
#         return snr

#     def compute_image_loss(self, mapping_feature, output_image, output_cond_image=None, cond_img_mask=None, output_image_feature=None):
#         if output_image_feature is not None:
#             latents = DiagonalGaussianDistribution(output_image_feature).sample()
#         else:
#             if len(output_image.shape) == 3:
#                 output_image = output_image.unsqueeze(0)

#             latents = self.vision_generator.vae.encode(output_image).latent_dist.sample()
#         if self.image_condition:
#             assert output_cond_image is not None, "the current model requires image as conditions"
#             # mask the uncond (can be accelerated here TODO!)
#             image_cond_latents = self.vision_generator.vae.encode(output_cond_image).latent_dist.mode()
#             cond_img_mask = cond_img_mask.to(image_cond_latents.dtype).unsqueeze(1).unsqueeze(2).unsqueeze(3)
#             image_cond_latents = cond_img_mask*image_cond_latents
        
#         latents = latents * self.vision_generator.vae.config.scaling_factor

#         noise = torch.randn_like(latents)
#         bsz = latents.shape[0]
#         # Sample a random timestep for each image
#         timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
#         timesteps = timesteps.long()

#         noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

#         target = noise

#         if self.image_condition:
#             # concatenate the image condition in the channels
#             noisy_latents = torch.cat([noisy_latents, image_cond_latents], dim=1)
#         unet_added_conditions = {}

#         if self.sd_add_args:
#             # time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(mapping_feature.device)
#             time_ids = torch.LongTensor([1024, 1024, 0, 0, 1024, 1024]).to(mapping_feature.device)
#             unet_added_conditions["time_ids"] = time_ids.repeat([bsz])
#             unet_added_conditions["text_embeds"] = torch.mean(mapping_feature, dim=1)
#         model_pred = self.vision_generator.unet(noisy_latents, 
#                                                 timesteps, mapping_feature,
#                                                 added_cond_kwargs=unet_added_conditions).sample


#         if self.config.snr_loss:
#             snr = self.compute_snr(timesteps)
#             mse_loss_weights = (
#                 torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
#             )
#             # We first calculate the original loss. Then we mean over the non-batch dimensions and
#             # rebalance the sample-wise losses with their respective loss weights.
#             # Finally, we take the mean of the rebalanced loss.
#             loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
#             loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
#             loss = loss.mean()
#         else:
#             loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
#         return loss
    
#     def encode_caption(self, caption, length, inference=False):
#         # add_special_tokens = False
#         # if len(caption) == 0:
#         add_special_tokens = True
#         text_inputs = self.vision_generator.sd_tokenizer(
#                 caption,
#                 padding="max_length",
#                 max_length=length,
#                 truncation=True,
#                 return_tensors="pt",
#                 add_special_tokens=add_special_tokens
#             ).to(self.device)
#         # text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
#         prompt_embeds = self.vision_generator.sd_text_encoder(**text_inputs)[0]
#         return prompt_embeds

#     def prepare_inputs_for_generation(
#         self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
#     ):
#         original_input_ids = input_ids.clone()
#         if past_key_values:
#             input_ids = input_ids[:, -1:]

#         # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#         if inputs_embeds is not None and past_key_values is None:
#             model_inputs = {"inputs_embeds": inputs_embeds}
#         else:
#             model_inputs = {"input_ids": input_ids}

#         model_inputs.update(
#             {
#                 "past_key_values": past_key_values,
#                 "use_cache": kwargs.get("use_cache"),
#                 "attention_mask": attention_mask,
#                 "input_images": kwargs.get("input_images", None),
#                 "box": kwargs.get("box", None),
#                 "image_label_masks": kwargs.get("image_label_masks", None),
#             }
#         )

#         if 'input_ids' in model_inputs:
#             new_token_ids = model_inputs['input_ids'][:, -1:]
#             if new_token_ids == self.boi_token:
#                 #Generated the image token, force add all the image tokens
#                 next_inputs_embeds, current_target_image_embeds = self.generate_image(model_inputs)
#                 self.to_generate_images.append(current_target_image_embeds)
#                 model_inputs['input_ids'] = None
#                 model_inputs['inputs_embeds'] = next_inputs_embeds
#                 all_img_tokens_mask = torch.ones(1, self.n_query).to(device=model_inputs['attention_mask'].device, dtype=model_inputs['attention_mask'].dtype)
#                 model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], all_img_tokens_mask], dim=1)
#             if new_token_ids == self.eoc_token_id and not self.no_bind:
#                 # need bounding box detection and use box align
#                 if self.sub_image_bind:
#                     next_inputs_embeds, query_len = self.generate_sub_image(model_inputs, original_input_ids)
#                 else:
#                     next_inputs_embeds, query_len = self.generate_box(model_inputs, original_input_ids)
#                 model_inputs['input_ids'] = None
#                 model_inputs['inputs_embeds'] = next_inputs_embeds
#                 all_img_tokens_mask = torch.ones(1, query_len).to(device=model_inputs['attention_mask'].device, dtype=model_inputs['attention_mask'].dtype)
#                 model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], all_img_tokens_mask], dim=1)
#         return model_inputs

#     def generate_box(self, model_inputs, original_input_ids):
#         assert original_input_ids.shape[0] == 1
#         if self.cache_images is None:
#             self.cache_images = self.encode_img(model_inputs['input_images'][0][:1])[0]
#         valid_start_ind = torch.where(original_input_ids==self.boc_token_id)[1].tolist()[-1]
#         current_box_text = self.tokenizer.decode(original_input_ids[0, valid_start_ind:])
#         current_box = torch.tensor(extract_box_str(current_box_text, mistral=True), dtype=self.dtype, device=self.cache_images.device)
#         if current_box is None:
#             print('fail to detect correct box from {}'.format(current_box_text))
#             raise ValueError
#         box_feat = self.box_align(self.cache_images[0], current_box.unsqueeze(0))[0]
#         init_inputs_embeds = self.get_input_embeddings()(model_inputs['input_ids'])
#         next_inputs_embeds = torch.cat([init_inputs_embeds, box_feat.unsqueeze(0)], dim=1)
#         return next_inputs_embeds, box_feat.shape[0]

#     def generate_sub_image(self, model_inputs, original_input_ids):
#         assert original_input_ids.shape[0] == 1
#         assert self.cache_raw_image is not None
#         valid_start_ind = torch.where(original_input_ids==self.boc_token_id)[1].tolist()[-1]
#         current_box_text = self.tokenizer.decode(original_input_ids[0, valid_start_ind:])
#         current_box = extract_box_str(current_box_text, mistral=True)
#         if current_box is None:
#             print('fail to detect correct box from {}'.format(current_box_text))
#             raise ValueError
#         # box_feat = self.box_align(self.cache_images[0], current_box.unsqueeze(0))[0]
#         w, h = self.cache_raw_image.size
#         x_min, y_min, x_max, y_max = current_box
#         x_min = x_min*w
#         y_min = y_min*h
#         x_max = x_max*w
#         y_max = y_max*h
#         sub_image = self.cache_raw_image.crop((x_min, y_min, x_max, y_max))
#         sub_image = self.image_processor(resize_image_to_square(sub_image)).unsqueeze(0).to(dtype=self.dtype, device=original_input_ids.device)
#         box_feat = self.encode_img(sub_image)[0][0]
#         init_inputs_embeds = self.get_input_embeddings()(model_inputs['input_ids'])
#         next_inputs_embeds = torch.cat([init_inputs_embeds, box_feat.unsqueeze(0)], dim=1)
#         return next_inputs_embeds, box_feat.shape[0]
    
#     def generate_image(self, model_inputs):
#         input_ids = model_inputs['input_ids']
#         past_key_values = model_inputs['past_key_values']
#         use_cache = model_inputs['use_cache']
#         attention_mask = model_inputs['attention_mask']
#         bs = input_ids.shape[0]
#         target_image_embeds = None
#         init_inputs_embeds = self.get_input_embeddings()(input_ids)
#         for num_img_token in range(self.n_query):
#             if num_img_token == 0:
#                 inputs_embeds = init_inputs_embeds
#             else:
#                 inputs_embeds = torch.cat([init_inputs_embeds, self.front_mm_projector(target_image_embeds)], dim=1)
#             if past_key_values is None:
#                 target_shape = num_img_token + 1
#             else:
#                 target_shape = past_key_values[-1][-1].shape[-2] + num_img_token + 1
#             attention_mask = torch.cat((attention_mask, torch.ones(
#                 (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
#                 dtype=attention_mask.dtype,
#                 device=attention_mask.device
#             )), dim=1)
#             sentence_length = torch.sum(attention_mask, dim=1).item()
#             position_ids = torch.arange(sentence_length-(num_img_token+1), sentence_length).unsqueeze(0).to(attention_mask.device)
#             outputs = self.model(
#                 input_ids=None,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 use_cache=use_cache,
#                 output_hidden_states=True,
#                 return_dict=True
#             )
#             hidden_states = outputs.hidden_states[-1]
#             target_image_embeds = hidden_states[:, -(num_img_token+1):, :]
#             target_image_embeds = self.behind_projector(target_image_embeds)
#         inputs_embeds = torch.cat([init_inputs_embeds, self.front_mm_projector(target_image_embeds)], dim=1)
#         return inputs_embeds, target_image_embeds
    
#     def flatten_hidden(self, hidden_state_list):
#         last_hidden_state = []
#         for hid_st in hidden_state_list:
#             last_hidden_state.append(hid_st[-1])
#         last_hidden_state = torch.cat(last_hidden_state,dim=1)
#         return last_hidden_state

#     def condition_completion(self, input_dict, temperature=0.2, max_new_tokens=128, guidance_scale=7.5, avoid_image_gen=False, **kwargs):
        
#         self.to_generate_images = []
#         self.cache_images = None
#         self.boi_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_BOI_TOKEN])[0]
#         if self.sub_image_bind:
#             self.cache_raw_image = input_dict['raw_images'][0][0]
#         else:
#             self.cache_raw_image = None
#         del(input_dict['raw_images'])
#         if len(input_dict['input_ids'].shape) == 1:
#             input_dict['input_ids'] = input_dict['input_ids'].unsqueeze(0)
#             input_dict['attention_mask'] = input_dict['attention_mask'].unsqueeze(0)
#         if isinstance(input_dict['input_images'], list):
#             input_dict['input_images'] = [item.to(self.dtype).to(self.device) if item is not None else item for item in input_dict['input_images']]
#         else:
#             input_dict['input_images'] = [input_dict['input_images'].to(self.dtype).to(self.device)] if input_dict['input_images'] is not None else [None]
#         with torch.no_grad():
#             text_out = self.generate(
#                         input_ids = input_dict['input_ids'].to(self.device), # TODO unsqueeze is for bs==1
#                         input_images=input_dict['input_images'] if input_dict['input_images'] is not None else [None],
#                         attention_mask = input_dict['attention_mask'].to(self.device),
#                         box = input_dict['box'] if 'box' in input_dict else None,
#                         do_sample = True if temperature > 0 else False,
#                         temperature=temperature,
#                         max_new_tokens = max_new_tokens,
#                         pad_token_id = self.tokenizer.pad_token_id,
#                         return_dict_in_generate = True
#                     )

#         input_token_len = input_dict['input_ids'].shape[1]
#         n_diff_input_output = (input_dict['input_ids'].to(self.device) != text_out.sequences[:, :input_token_len]).sum().item()
#         if n_diff_input_output > 0:
#             print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
#         pred_out = self.tokenizer.batch_decode(text_out.sequences[:, input_token_len:], skip_special_tokens=False)
        
#         # check if images are in the output, if so, decode them
#         output_images = []
#         if len(self.to_generate_images) and not avoid_image_gen:
#             self.vision_generator.image_pipeline.to(self.device, self.dtype)
#             negative_embeds = self.encode_img(torch.zeros(1, 3, self.vision_encoder.image_size, self.vision_encoder.image_size).to(device=self.device, dtype=self.dtype))[2]
#             for to_gen_img in self.to_generate_images:
#                 out_img = self.vision_generator.image_pipeline(prompt_embeds=to_gen_img, negative_embeds=negative_embeds, guidance_scale=3, height=1024, width=1024, crop_info=[0, 0], original_size=[1024, 1024], num_inference_steps=100).image         
#                 output_images.append(out_img)
#         return pred_out, output_images, text_out.sequences

#     def calculate_options(self, input_dict, cot=False, further_instruct=False, temperature=0.2, max_new_tokens=128, guidance_scale=7.5, avoid_image_gen=False, likelihood_reduction='mean', **kwargs):
        
#         assert len(input_dict['options']) == 1
#         self.cache_images = None
#         self.boi_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_BOI_TOKEN])[0]
#         options = [s for s in input_dict['options'][0]]
#         del(input_dict['options'])
#         if self.sub_image_bind:
#             self.cache_raw_image = input_dict['raw_images'][0][0]
#         else:
#             self.cache_raw_image = None
#         del(input_dict['raw_images'])
#         if len(input_dict['input_ids'].shape) == 1:
#             input_dict['input_ids'] = input_dict['input_ids'].unsqueeze(0)
#             input_dict['attention_mask'] = input_dict['attention_mask'].unsqueeze(0)
#         if isinstance(input_dict['input_images'], list):
#             input_dict['input_images'] = [item.to(self.dtype).to(self.device) if item is not None else item for item in input_dict['input_images']]
#         else:
#             input_dict['input_images'] = [input_dict['input_images'].to(self.dtype).to(self.device)] if input_dict['input_images'] is not None else [None]
#         if cot:
#             # need to first conduct the thinking
#             with torch.no_grad():
#                 text_out = self.generate(
#                             input_ids = input_dict['input_ids'].to(self.device), # TODO unsqueeze is for bs==1
#                             input_images=input_dict['input_images'] if input_dict['input_images'] is not None else [None],
#                             attention_mask = input_dict['attention_mask'].to(self.device),
#                             box = input_dict['box'] if 'box' in input_dict else None,
#                             do_sample = True if temperature > 0 else False,
#                             temperature=temperature,
#                             max_new_tokens = max_new_tokens,
#                             pad_token_id = self.tokenizer.pad_token_id,
#                             return_dict_in_generate = True
#                         )
#             thought_ids = text_out.sequences
#             input_token_len = input_dict['input_ids'].shape[1]
#             n_diff_input_output = (input_dict['input_ids'].to(self.device) != text_out.sequences[:, :input_token_len]).sum().item()
#             if n_diff_input_output > 0:
#                 print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
#             thought = self.tokenizer.batch_decode(text_out.sequences[:, input_token_len:], skip_special_tokens=False)[0]
#             thought_ids = thought_ids.squeeze()
#             if further_instruct:
#                 # need to instruct the model to select from options!
#                 options_instruct = 'Select from following options: ' + '; '.join(options) + '.'
#                 option_instruct_ids = self.tokenizer([options_instruct], return_tensors='pt')['input_ids'][:, 1:].squeeze()
#                 suffix = torch.cat([torch.tensor([733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804]), 
#                                     option_instruct_ids, 
#                                     torch.tensor([733, 28748, 16289, 28793])]).to(thought_ids.device)
#             else:
#                 suffix = torch.tensor([  733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
#                                         28748, 16289, 28793]).to(thought_ids.device)
#             eoc_indices = [-1] + torch.where(thought_ids == self.eoc_token_id)[0].tolist() + [thought_ids.shape[0]-1]
#             input_ids = []
#             for i in range(len(eoc_indices) - 1):
#                 input_ids.append(thought_ids[eoc_indices[i]+1:eoc_indices[i+1]+1])
#                 if i < len(eoc_indices) - 2:
#                     if thought_ids[eoc_indices[i+1]+1].item() != self.input_img_id:
#                         input_ids.append(torch.tensor([self.input_img_id]).to(thought_ids.device))
#             input_ids.append(suffix)
#             input_ids = torch.cat(input_ids).unsqueeze(0)
#             new_thought, thought_boxes = process_thought(thought, mistral=True)
#             if self.sub_image_bind:
#                 w, h = self.cache_raw_image.size
#                 new_sub_images = []
#                 for b in thought_boxes:
#                     x_min, y_min, x_max, y_max = b
#                     x_min = x_min*w
#                     y_min = y_min*h
#                     x_max = x_max*w
#                     y_max = y_max*h
#                     sub_image = self.cache_raw_image.crop((x_min, y_min, x_max, y_max))
#                     new_sub_images.append(self.image_processor(resize_image_to_square(sub_image)).unsqueeze(0).to(dtype=self.dtype, device=input_ids.device))
#                 num_images = len(new_sub_images) + 1
#                 input_dict['input_images'] = [torch.cat(input_dict['input_images'] + new_sub_images, dim=0)]
#                 all_box = [torch.tensor([[0.0, 0.0, 1.0, 1.0]]*num_images, device=input_ids.device, dtype=input_dict['box'][0].dtype)]
#             else: 
#                 all_box = input_dict['box'][0]
#                 all_box = [torch.cat([all_box, torch.tensor(thought_boxes, device=all_box.device, dtype=all_box.dtype)], dim=0)]
#         else:
#             input_ids = input_dict['input_ids']
#             all_box = input_dict['box']
        
#         # calculate the past qk for processing
#         input_dict['input_ids'] = input_ids.to(self.device)
#         input_dict['box'] = all_box
#         input_dict['use_cache'] = True
#         input_dict['attention_mask'] = torch.ones_like(input_dict['input_ids']).to(self.device)
#         del(input_dict['labels'])
#         question_output = self.forward(**input_dict)
#         question_logits = question_output.logits

#         # calculate the logit
#         option_losses = []
#         for opt in options:
#             opt_ids = self.tokenizer([opt], return_tensors='pt')['input_ids'][:, 1:].to(self.device)
#             opt_output = self.forward(input_ids = opt_ids, attention_mask=torch.ones_like(opt_ids), 
#                                     past_key_values=question_output.past_key_values, use_cache=True)
#             logits = torch.cat([question_logits[:, -1:], opt_output.logits[:, :-1]], 1)
#             if likelihood_reduction == 'mean':
#                 loss_fct = CrossEntropyLoss()
#                 logits = logits.view(-1, self.config.vocab_size)
#                 labels = opt_ids.view(-1)
#                 loss = loss_fct(logits, labels)
#             elif likelihood_reduction == 'sum':
#                 loss_fct = CrossEntropyLoss(reduction='none')
#                 logits = logits.view(-1, self.config.vocab_size)
#                 labels = opt_ids.view(-1)
#                 loss = loss_fct(logits, labels).sum()
#             else:
#                 raise ValueError
#             option_losses.append(loss)
        
#         return torch.stack(option_losses).argmin().cpu().item(), thought if cot else None

#     def load_state_dict_from_ckpt(self,ckpt_file):
#         state_dict = torch.load(ckpt_file)
#         if 'state_dict' in state_dict:
#             state_dict = state_dict['state_dict']
        
#         new_state_dict = dict()
#         for key in state_dict.keys():
#             if 't2i_decoder_prompt' in key or 'llm_to_t2i_mapping' in key:
#                 new_key = 'behind_projector.'+key
#             elif 'llama_model' in key:
#                 if 'lora' in key or 'modules_to_save' in key:
#                     new_key = '.'.join(key.split('.')[4:])
#                 else:
#                     new_key = '.'.join(key.split('.')[2:])
#             new_state_dict[new_key] = state_dict[key]

#         model_state_dict = self.state_dict()
#         model_state_dict.update(new_state_dict)
#         self.load_state_dict(model_state_dict)

#     def load_state_dict_from_old_code_checkpoint(self,ckpt_file):
#         state_dict = torch.load(ckpt_file)
#         prefix_key = list(set(['.'.join(key.split('.')[:2]) for key in state_dict.keys()]))
#         new_state_dict = dict()
#         for key in state_dict.keys():
#             if 't2i_decoder_prompt' in key or 'llm_to_t2i_mapping' in key:
#                 new_key = 'behind_projector.'+key
#             elif 'llama_model' in key:
#                 new_key = '.'.join(key.split('.')[4:])
#             elif 'vae' in key or 'unet' in key or 'sd_text_encoder' in key:
#                 new_key = 'vision_generator.'+key
#                 # check the unet condition conv_in layer
#                 if 'unet.conv_in.weight' == key:
#                     num_channels = state_dict[key].shape[1]
#                     if self.image_condition:
#                         if self.config.vision_generator_cond_channels + 4 != num_channels:
#                             continue
#             elif 'Qformer' in key:
#                 new_key = 'front_mm_projector.'+'.'.join(key.split('.')[2:])
#             elif 'query_tokens' in key or 'llama_proj' in key:
#                 new_key = '.'.join(key.split('.')[1:])
#             elif 'ln_vision' in key:
#                 new_key = 'vit_ln.' + '.'.join(key.split('.')[1:])
#             elif 'visual_encoder' in key:
#                 new_key = 'vision_encoder.'+ '.'.join(key.split('.')[2:])
#             elif 'fc' in key:
#                 new_key = key
#             else:
#                 raise ValueError('no support key from old code checkpoint')

#             new_state_dict[new_key] = state_dict[key]

#         self.load_state_dict(new_state_dict, strict=False)
# AutoConfig.register("VolCanoMistral", VolCanoMistralConfig)
# AutoModelForCausalLM.register(VolCanoMistralConfig, VolCanoMistralForCausalLM)

def nonzero_mean(x, axis=None):
    if axis is not None:
        return x.sum(axis) / (x != 0).sum(axis)
    return x.sum() / (x != 0).sum()

def loss_mean(x):
    return x.sum() / (x != 0).sum()
class QuietVolCanoMistralForCausalLM(MistralForCausalLM, VolCanoMetaForCausalLM):
    config_class = VolCanoMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        
        self.model = VolCanoMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_vision_model()

        if hasattr(self, 'vision_generator'):

            zero_img_feature = torch.zeros((1, IMG_TOKEN_NUM, config.hidden_size))
            self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

        # Initialize weights and apply final processing
        # self.post_init()       # 重复
        self.output_img_id = -100
        self.regression_weight = 1.0
        self.no_bind = False
        
        #################### quiet thought #####################
        # self.model = MistralModel(config)   # 重复
        self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 重复
        self.max_thoughts = config.max_thoughts
        self.merged_lm_and_talk_heads = config.merged_lm_and_talk_heads
        self.use_concat_talk_head = config.use_concat_talk_head
        self.use_shallow_talk = config.use_shallow_talk
        self.use_complex_talk_head = config.use_complex_talk_head
        self.use_weighted_talk_head = config.use_weighted_talk_head
        # the weighted head will output a single value, so it can't be passed to the lm head
        assert not (self.use_weighted_talk_head and self.use_shallow_talk), (
        "Both `use_weighted_talk_head` and `use_shallow_talk` cannot be True simultaneously.")

        self.n_ahead = 1
        self.n_ahead_talk = 1
        self.n_passes = 1
        self.n_tokens_print = 1
        self.gradient_accumulation_steps = 1
        self.training_steps = 0
        self.tokenizer = None
        self.start_token_id = None
        self.end_token_id = None
        self.rm_initialized = False
        self.residual_talk_head = True
        self.thought_init_std_scale = 1e-2

        self.final_only_mode = False
        self.first_and_last_mode = True
        self.first_only = False
        self.original_loss_weight = 0.5

        self.cumulative_residual = False
        self.clever_residual = False
        self.skip_residual = False
        self.no_residual = True

        self.optimize_lm_head_only_at_start = False
        self.optimize_model_only_at_start = False

        if self.optimize_model_only_at_start:
            raise NotImplementedError
        self.train_only_thinking_embedding = False
        self.weighted_embeddings = False
        self.use_start_thought_token = True
        self.use_end_thought_token = True
        self.initialize_thought_embedding_to_normal = False
        self.initial_start_token = "---"
        self.initial_end_token = "---"
        self.output_logits_at_the_end = True

        self.wandb_enabled = False
        self.gumbel_temperature = 0.001

        self.use_policy_loss = True
        self.include_policy_loss = True
        self.trice_mode = True
        self.remove_negative_rewards = True
        self.use_policy_loss_for_end_thought = True
        
        self.base_original_mode = False
        self.original_mode = False

        self.thought_prefix = "(Let's think step by step"
        self.tokenized_thought_prefix = None
        self.log_dict = defaultdict(int)
        self.eval_log_dict = defaultdict(int)
        self.print_final_only = True
        self.loss_mean = loss_mean
        self.all_rewards = []
        self.all_unreduced_losses = []
        self.kill_after = 100

        self.start_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))
        self.end_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))
 
        self.policy_loss_beta = 1e6
        self.embedding_scale = 1e2
        self.reinforce_temperature = 3
        self.base_loss_beta = 1

        # Not used in the paper:
        self.use_thought_prefix = False
        self.use_reparam_for_thought_embeddings = False
        self.use_upper_triangular = False
        self.subtract_mean_reward = False
        self.comparison_mode = False
        self.gumbel_detach = True
    
        # For visualization
        self.eval_mode = False

        num_talk = 1
        talk_input_dim = config.hidden_size if not self.use_concat_talk_head else config.hidden_size * 2
        if self.use_weighted_talk_head:
            talk_output_dim = 1
        else:
            talk_output_dim = config.hidden_size if self.use_shallow_talk else config.vocab_size

        if not self.merged_lm_and_talk_heads:
            if self.use_complex_talk_head:
                self.talk_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(talk_input_dim, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, talk_output_dim, bias=False)
                )])
            else:
                self.talk_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                )])

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(    
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        captions=[None],
        output_image_feature=None,
        output_images = None,
        output_cond_images = None,
        output_cond_img_mask = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_images: Optional[torch.FloatTensor] = None,
        image_label_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        item_id: Optional[bool] = None,
        box: Optional[torch.Tensor] = None 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # return torch.zeros(1).to(self.device).to(self.dtype)
        ################# vocot处理输入 ########   
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, visual_labels, visual_label_masks = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids ,attention_mask, past_key_values, labels = labels, images = input_images, image_label_masks=image_label_masks, inputs_embeds=inputs_embeds, box=box)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.dtype)  
        
        # 为了和quietstar对齐,    
        attention_mask = None
        ####################################################################
        
        
        log_dict = self.log_dict if self.training else self.eval_log_dict

        if self.training and self.kill_after is not None and self.training_steps // self.gradient_accumulation_steps > self.kill_after:
            raise ValueError("Killed after")

        if not self.training:
            n_ahead_talk_to_restore = self.n_ahead_talk
            n_passes_to_restore = self.n_passes
            self.n_ahead_talk = 1
            self.n_passes = 1

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert self.cumulative_residual or self.clever_residual or self.skip_residual or self.no_residual
        assert not (self.skip_residual and self.use_policy_loss)

        if self.tokenized_thought_prefix is None and self.use_thought_prefix:
            self.tokenized_thought_prefix = self.tokenizer(self.thought_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
            # self.tokenized_thought_prefix 处理成 input_ids的形式
        def apply_head(head, states, detach=False):
            if detach:
                head_weight = head.weight.detach()
            else:
                head_weight = head.weight
            head_weight = head_weight.to(states.device)
            return (head_weight @ states.transpose(-1, -2)).transpose(-1, -2).contiguous()
    
        def idx_if_sequential(head, idx=0):
            if isinstance(head, nn.Sequential) or isinstance(head, nn.ModuleList):
                return idx_if_sequential(head[idx], idx=idx)
            return head

        def none_repeat_interleave(x, n):
            if x is None:
                return x
            return x.repeat_interleave(n, dim=0)

        if self.n_passes > 1:
            # inputs_embeds = none_repeat_interleave(inputs_embeds, self.n_passes)
            attention_mask = none_repeat_interleave(attention_mask, self.n_passes)
            position_ids = none_repeat_interleave(position_ids, self.n_passes)
            inputs_embeds = none_repeat_interleave(inputs_embeds, self.n_passes)
            labels = none_repeat_interleave(labels, self.n_passes)
            if past_key_values is not None:
                past_key_values = [none_repeat_interleave(p, self.n_passes) for p in past_key_values]
        cur_token_indices = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        self.tokenizer_has_start_thought_token = True
        self.tokenizer_has_end_thought_token = True
        if self.start_token_id is None:
            self.start_token_id = self.tokenizer.convert_tokens_to_ids("<|startthought|>")
            if self.start_token_id == 0:
                self.start_token_id = self.tokenizer.bos_token_id
                self.tokenizer_has_start_thought_token = False
            elif self.use_start_thought_token:
                # base_start_id = self.tokenizer.convert_tokens_to_ids(self.initial_start_token)
                base_start_id = self.tokenizer.encode(self.initial_start_token, add_special_tokens=False)[0]
                if self.initialize_thought_embedding_to_normal:
                    self.start_embedding.data = torch.zeros_like(self.start_embedding.data)
                else:
                    self.start_embedding.data[0] = self.model.embed_tokens.weight.data[base_start_id].clone().detach() / self.embedding_scale
                self.start_embedding.data[1] = torch.log(self.model.embed_tokens.weight.data.std(dim=0) * self.thought_init_std_scale / self.embedding_scale)
        if self.end_token_id is None:
            self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|endthought|>")
            if self.end_token_id == 0:
                self.end_token_id = self.tokenizer.eos_token_id
                self.tokenizer_has_end_thought_token = False
            elif self.use_end_thought_token:
                # base_end_id = self.tokenizer.convert_tokens_to_ids(self.initial_end_token)
                base_end_id = self.tokenizer.encode(self.initial_end_token, add_special_tokens=False)[0]
                if self.initialize_thought_embedding_to_normal:
                    self.end_embedding.data = torch.zeros_like(self.end_embedding.data)
                else:
                    self.end_embedding.data[0] = self.model.embed_tokens.weight.data[base_end_id].clone().detach() / self.embedding_scale
                self.end_embedding.data[1] = torch.log(self.model.embed_tokens.weight.data.std(dim=0) * self.thought_init_std_scale / self.embedding_scale)

        if not self.rm_initialized and (self.n_ahead > 1 or not self.base_original_mode):
            self.rm_initialized = True                        
            if not self.use_shallow_talk:
                head = self.talk_head[0]
                cur_head = head[-1] if isinstance(head, nn.Sequential) else head
                talk_input_dim = cur_head.weight.data.shape[1]
                talk_output_dim = 1 if self.use_weighted_talk_head else self.lm_head.weight.data.shape[0]
                cur_head.weight.data = torch.zeros(talk_output_dim, talk_input_dim, device=cur_head.weight.device, dtype=cur_head.weight.dtype)
            else:
                # convert to identity transform
                def lambda_transform(cur_head):
                    if cur_head.weight.data.shape[0] != cur_head.weight.data.shape[1]:
                        return torch.cat([
                        torch.eye(
                            cur_head.weight.data.shape[0],
                            device=cur_head.weight.device,
                            dtype=cur_head.weight.dtype
                        ),
                        torch.zeros(
                            cur_head.weight.data.shape[0],
                            cur_head.weight.data.shape[1] - cur_head.weight.data.shape[0],
                            device=cur_head.weight.device,
                            dtype=cur_head.weight.dtype
                        )], dim=1)
                    return torch.eye(
                        cur_head.weight.data.shape[0],
                        device=cur_head.weight.device,
                        dtype=cur_head.weight.dtype
                    )
                if isinstance(self.talk_head[0], nn.Sequential):
                    for cur_head in self.talk_head[0]:
                        # if it has weights
                        if hasattr(cur_head, "weight"):
                            cur_head.weight.data = lambda_transform(cur_head)
                else:
                    self.talk_head[-1].weight.data = lambda_transform(self.talk_head[0])

        loss = None
        prev_rm_tokens = None
        cur_rm_tokens = None
        prev_rm_logits = None
        prev_sample_probs = None
        did_skip_sampling = None
        skip_sampling = None
        sample_probs = None
        hidden_states = None
        logits = None
        talk_kl_penalty = None
        rm_logits = None
        residual_logits = None
        probabilities_2d = None
        prev_probabilities_2d = None
        policy_reward = None
        logits_to_output = None
        batch_size, seq_len,_ = inputs_embeds.shape
        base_inputs_embeds = inputs_embeds.clone()
        loss_list = []
        dqn_loss_list = []
        sampled_token_history = []
        sample_probs_history = []
        action_loglikelihoods_list = []

        if self.use_end_thought_token or self.use_start_thought_token:
            if not self.use_reparam_for_thought_embeddings:
                start_embedding = self.start_embedding[0].unsqueeze(0) * self.embedding_scale
                end_embedding = self.end_embedding[0].unsqueeze(0) * self.embedding_scale
            else:
                start_embedding = self.start_embedding * self.embedding_scale
                end_embedding = self.end_embedding * self.embedding_scale
            base_embeddings = self.model.embed_tokens.weight
            if self.train_only_thinking_embedding:
                base_embeddings = base_embeddings.detach()
        # # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        fwd_iters = 1 if self.original_mode else self.n_ahead + self.n_ahead_talk - 1
        for ahead_idx in range(fwd_iters):
            past_key_values_length = 0
            if past_key_values is not None:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_len)

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
            else:
                position_ids = position_ids.view(-1, seq_len).long()

            if inputs_embeds is None: # 说明使用的是 input_ids
                contains_start = self.use_start_thought_token and (input_ids == self.start_token_id).any()
                contains_end = self.use_end_thought_token and (input_ids == self.end_token_id).any()
                contains_thought = contains_start or contains_end
                if contains_thought:
                    thought_id = self.start_token_id if contains_start else self.end_token_id
                    cur_thought_embedding = start_embedding if contains_start else end_embedding
                    if self.use_reparam_for_thought_embeddings:
                        inputs_embeds = torch.randn(batch_size, seq_len, self.model.config.hidden_size, device=input_ids.device, dtype=cur_thought_embedding.dtype)
                        inputs_embeds = inputs_embeds.detach() * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
                        if contains_start:
                            sampled_start = inputs_embeds.clone().detach()
                        if contains_end:
                            sampled_end = inputs_embeds.clone().detach()
                    else:
                        inputs_embeds = cur_thought_embedding.unsqueeze(0).repeat(batch_size, seq_len, 1)
                else:
                    with torch.set_grad_enabled(not self.train_only_thinking_embedding):
                        inputs_embeds = self.model.embed_tokens(input_ids)
            
            if self.n_ahead != 1 or self.n_ahead_talk != 1 or self.comparison_mode:
                if attention_mask is None:
                    base_attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(inputs_embeds.device)
                    base_attention_mask = base_attention_mask.view(1, 1, seq_len, seq_len)
                    base_attention_mask = base_attention_mask.repeat(inputs_embeds.shape[0], 1, 1, 1)
                    attention_mask = base_attention_mask
                    breakpoint()
                elif attention_mask.dim() == 2:
                    if seq_len + past_key_values_length != attention_mask.shape[-1]:
                        breakpoint()
                        attention_mask = torch.cat(
                            [torch.ones((attention_mask.shape[0], past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask],
                            dim=-1
                        )
                    # # if the attention mask 
                    attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask,
                        (batch_size, seq_len),
                        inputs_embeds,
                        past_key_values_length,
                        sliding_window=self.config.sliding_window,
                    )

            outputs = self.model(
                # input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            prev_hidden_states = hidden_states
            hidden_states = outputs[0]
            prev_rm_logits = rm_logits  # for policy gradient
            prev_rm_tokens = cur_rm_tokens  # for policy gradient

            if ahead_idx == 0:
                hidden_states_lm = hidden_states
                logits = self.lm_head(hidden_states_lm)
                base_hidden_states = hidden_states.clone()
                initial_loss_logits = logits.clone()
                if self.optimize_lm_head_only_at_start or self.optimize_model_only_at_start:
                    logits = logits.detach()
                    base_hidden_states = base_hidden_states.detach()
                if self.optimize_model_only_at_start:
                    hidden_states = hidden_states.detach()
                base_logits = logits.clone()
            else:
                talk_hidden_states = hidden_states
                if self.merged_lm_and_talk_heads:
                    assert self.no_residual
                    residual_logits = self.lm_head(hidden_states)
                    talk_hidden_states = hidden_states
                else:
                    if ahead_idx > self.n_ahead - 1:
                        cur_base_hidden = torch.cat([
                            base_hidden_states[..., ahead_idx - self.n_ahead + 1:, :],
                            base_hidden_states[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                    else:
                        cur_base_hidden = base_hidden_states

                    if self.use_concat_talk_head:
                        # concatenate the hidden states with the original hidden states
                        head_input_hidden_states = torch.cat([cur_base_hidden, talk_hidden_states], dim=-1)
                    else:
                        head_input_hidden_states = talk_hidden_states

                    residual_logits = self.talk_head[0](head_input_hidden_states)
                    if self.use_shallow_talk:
                        residual_logits = apply_head(self.lm_head, residual_logits, detach=self.optimize_lm_head_only_at_start)                        
                    residual_logits = residual_logits.to(logits.device)
                    if self.use_weighted_talk_head:
                        # combine the cur_base_hidden with the talk_hidden_states according to the weighted head
                        residual_logits = cur_base_hidden * (1 - residual_logits) + talk_hidden_states * residual_logits
                        residual_logits = apply_head(self.lm_head, residual_logits, detach=self.optimize_lm_head_only_at_start)

                assert sum([self.cumulative_residual, self.clever_residual, self.skip_residual, self.no_residual]) == 1
                if self.clever_residual:
                    if ahead_idx >= self.n_ahead - 1:
                        # get the logits shifted according to the current talk ahead
                        cur_base_logits = torch.cat([
                            base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                            base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                        if self.optimize_lm_head_only_at_start:
                            cur_base_logits = cur_base_logits.detach()
                        logits = cur_base_logits + residual_logits
                    else:
                        logits += residual_logits / self.n_ahead
                elif self.cumulative_residual:
                    if self.residual_talk_head:
                        if ahead_idx < self.n_ahead:
                            logits += residual_logits
                        else:
                            # get the logits shifted according to the current talk ahead
                            cur_base_logits = torch.cat([
                                base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                                base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                            ], dim=-2)
                            if self.optimize_lm_head_only_at_start:
                                cur_base_logits = cur_base_logits.detach()
                            logits = cur_base_logits + residual_logits
                    else:
                        if ahead_idx < self.n_ahead:
                            logits += residual_logits
                        else:
                            logits = residual_logits
                elif self.skip_residual:
                    if ahead_idx >= self.n_ahead:
                        # get the logits shifted according to the current talk ahead
                        cur_base_logits = torch.cat([
                            base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                            base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                        if self.optimize_lm_head_only_at_start:
                            cur_base_logits = cur_base_logits.detach()
                        logits = cur_base_logits
                elif self.no_residual:
                    logits = residual_logits
                else:
                    logits = base_logits + residual_logits

            attempted = False
            talk_loss_list = []
            if self.original_mode or (self.n_ahead == 1) or (self.comparison_mode and ahead_idx == 0):# or (self.optimize_lm_head_only_at_start and ahead_idx == 0):
                loss = None
                attempted = True

                if labels is not None:
                    for shift_amount in range(self.n_ahead_talk):
                        # Shift so that tokens < n predict n
                        #  ab[cde]f
                        # abc[def]
                        if ahead_idx == 0 and self.optimize_lm_head_only_at_start:
                            loss_logits = initial_loss_logits
                        else:
                            loss_logits = logits
                        shift_logits = loss_logits[..., shift_amount:-1, :].contiguous()
                        shift_labels = labels[..., 1 + shift_amount:].contiguous()
                        # Flatten the tokens
                        loss_fct = CrossEntropyLoss(reduction="none")
                        shift_logits = shift_logits.view(-1, self.config.vocab_size)
                        shift_labels = shift_labels.view(-1).clone()
                        # Enable model parallelism
                        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
                        shift_labels = shift_labels.to(shift_logits.device)
                        loss = loss_fct(shift_logits, shift_labels)
                        if not self.comparison_mode and not (self.optimize_lm_head_only_at_start and (self.n_ahead + self.n_ahead_talk > 2)) or self.original_mode:
                            loss_list.append(loss)
                        talk_loss_list.append(nonzero_mean(loss).detach())
            
            if not attempted or self.comparison_mode:
                rm_hidden_states = hidden_states
                # print("Magnitude of RM hidden states before RM head", rm_hidden_states.norm())
                rm_logits = apply_head(self.lm_head, rm_hidden_states, detach=self.optimize_lm_head_only_at_start)
                    
                # don't allow it to predict the thinking token
                if self.tokenizer_has_start_thought_token:                    
                    rm_logits[..., self.start_token_id] = -1e10
                if self.tokenizer_has_end_thought_token:
                    rm_logits[..., self.end_token_id] = -1e10
                probabilities = rm_logits
                if probabilities_2d is not None:
                    prev_probabilities_2d = probabilities_2d.clone()
                probabilities_2d = probabilities.view(-1, probabilities.size(-1))

                did_skip_sampling = skip_sampling
                skip_sampling = False
                if ahead_idx == 0 and self.use_start_thought_token:
                    override_token = self.start_token_id
                elif self.use_thought_prefix and ahead_idx < self.tokenized_thought_prefix.shape[-1]:
                    override_token = self.tokenized_thought_prefix[..., ahead_idx]
                elif ahead_idx == self.n_ahead - 2 and self.use_end_thought_token:
                    override_token = self.end_token_id
                else:
                    override_token = None
                if override_token is not None and self.n_ahead > 1:
                    # always start with the start token
                    probabilities_2d = torch.zeros_like(probabilities_2d)
                    probabilities_2d[:, override_token] = 1.0
                    skip_sampling = True
                elif ahead_idx >= self.n_ahead - 1:
                    if labels is not None:  # we're in the talk phase
                        cur_talk_n = ahead_idx - (self.n_ahead - 1) + 1
                        # print("Setting rm to labels", cur_talk_n, "during", ahead_idx)
                        shift_labels = labels[..., cur_talk_n:].contiguous().to(probabilities_2d.device)
                        padding = torch.full_like(
                            labels[..., :cur_talk_n],
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=shift_labels.device
                        )
                        new_rm_tokens = torch.cat(
                            [shift_labels, padding],
                            dim=-1
                        )
                        # convert rm tokens to one-hot
                        probabilities_2d = F.one_hot(new_rm_tokens, num_classes=self.vocab_size).reshape(-1, self.vocab_size).to(probabilities_2d.dtype)
                        skip_sampling = True
                    else:
                        continue
                temperature = self.gumbel_temperature if self.training else 0.001
                prev_sample_probs = sample_probs
                sample_probs = probabilities_2d
                if ahead_idx < self.n_ahead - 1 and not skip_sampling:
                    probabilities_2d = F.gumbel_softmax(sample_probs, tau=temperature, hard=True, dim=-1)
                    if self.gumbel_detach:
                        probabilities_2d = probabilities_2d.detach()
                sampled_token_history.append(probabilities_2d.argmax(dim=-1).detach().cpu())
                # convert rm logits directly to embeddings
                contains_start = self.use_start_thought_token and (probabilities_2d[..., self.start_token_id].sum() > 0)
                contains_end = self.use_end_thought_token and (probabilities_2d[..., self.end_token_id].sum() > 0)
                contains_thought = contains_start or contains_end

                if not contains_thought:
                    with torch.set_grad_enabled(not self.train_only_thinking_embedding):
                        inputs_embeds = probabilities_2d @ (self.model.embed_tokens.weight.to(probabilities.device).to(probabilities.dtype))
                else:
                    thought_id = self.start_token_id if contains_start else self.end_token_id
                    cur_thought_embedding = start_embedding if contains_start else end_embedding
                    if self.use_reparam_for_thought_embeddings:
                        inputs_embeds = torch.randn(batch_size, seq_len, self.model.config.hidden_size, device=inputs_embeds.device, dtype=cur_thought_embedding.dtype)
                        inputs_embeds = inputs_embeds * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
                        if contains_start:
                            sampled_start = inputs_embeds.clone().detach()
                        else:
                            sampled_end = inputs_embeds.clone().detach()
                    else:
                        inputs_embeds = cur_thought_embedding.unsqueeze(0).repeat(batch_size, seq_len, 1)
                        inputs_embeds = inputs_embeds.view(probabilities.size(0), probabilities.size(1), -1).to(self.model.embed_tokens.weight.dtype)
                inputs_embeds = inputs_embeds.view(probabilities.size(0), probabilities.size(1), -1).to(self.model.embed_tokens.weight.dtype)

                if len(attention_mask.shape) == 2:
                    breakpoint()
                else:
                    original_attention = attention_mask[..., :attention_mask.shape[-2]]
                    if self.use_upper_triangular:
                        new_attention = original_attention
                    else:
                        original_attention = original_attention == attention_mask.max()
                        # because eye isn't implemented for BF16, we need to handle the case
                        if not attention_mask.dtype == torch.bfloat16:
                            new_attention = torch.eye(
                                seq_len, dtype=attention_mask.dtype, device=attention_mask.device
                            )
                        else:
                            new_attention = torch.eye(
                                seq_len, dtype=torch.float32, device=attention_mask.device
                            ).to(attention_mask.dtype)

                        new_attention = new_attention.view(1, 1, seq_len, seq_len).repeat(inputs_embeds.shape[0], 1, 1, 1)
                        new_attention = new_attention * original_attention
                        new_attention[new_attention == 0] = attention_mask.min()
                        new_attention[new_attention == 1] = attention_mask.max()
                    attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
                past_key_values = outputs.past_key_values
                position_ids = position_ids + 1

                if labels is not None and (self.n_ahead > 1 or not self.base_original_mode):
                    # Shift so that tokens < n predict n
                    # logits: abcdef -> bcdef? -> cdef??
                    # labels: abcdef -> ?bcdef -> ??cdef
                    if ahead_idx == 0 and self.optimize_lm_head_only_at_start:
                        loss_logits = initial_loss_logits
                    else:
                        loss_logits = logits
                    shift_idx = 1 + max(0, ahead_idx - (self.n_ahead - 1))
                    shift_logits = loss_logits[..., :-shift_idx, :].contiguous()
                    shift_labels = labels[..., shift_idx:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduction="none")
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    # if shift_labels.min() == self.tokenizer.pad_token_id:
                    shift_labels = torch.where(shift_labels == self.tokenizer.pad_token_id, -100, shift_labels)
                    unreduced_loss = loss_fct(shift_logits, shift_labels)
                    if torch.any(unreduced_loss != unreduced_loss):
                        raise ValueError("NaN loss")
                    unreduced_loss = unreduced_loss.reshape(logits.shape[0], -1)
                    loss_list.append(unreduced_loss)


                    if self.use_policy_loss and ahead_idx > 0 and (ahead_idx > 1 or not self.use_start_thought_token):
                        # we treat the change in loss as the reward
                        previous_loss = loss_list[-2]
                        # for example, suppose n_ahead = 3 and n_ahead_talk = 2
                        # note that we end at self.n_ahead + self.n_ahead_talk - 2
                        # in this case, 5 - 2 = 3, so we end at ahead_idx = 3
                        # we also predict the next token at ahead_idx = 2
                        # when we get to ahead_idx = 2, we predict ahead
                        # so we shift by 1
                        # note that this is ahead_idx = n_ahead - 1
                        # when we get to ahead_idx = 3, we predict ahead
                        # so we shift by 2
                        # note that this is ahead_idx = n_ahead
                        if ahead_idx < self.n_ahead - 1:
                            shift_amount = 0
                            original_dqn_reward = (previous_loss - unreduced_loss).detach()
                            if self.first_and_last_mode:
                                original_dqn_reward = original_dqn_reward * 0.0
                        else:
                            # logits vs cur_policy_shift_logits
                            # let's look at rm_logits and prev_rm_logits
                            shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                            # let's say shift_amount = 2
                            # abcdefg -> bcdefg? -> cdefg??
                            # logits = [a b]c d e f[g]
                            # labels = [a b c]d e f g
                            cur_policy_shift_logits = initial_loss_logits[..., shift_amount:-1, :].contiguous().detach()
                            cur_policy_shift_labels = labels[..., 1 + shift_amount:].contiguous()
                            # Flatten the tokens
                            cur_policy_loss_fct = CrossEntropyLoss(reduction="none")
                            cur_policy_shift_logits = cur_policy_shift_logits.view(-1, self.config.vocab_size)
                            cur_policy_shift_labels = cur_policy_shift_labels.view(-1).clone()
                            # Enable model parallelism
                            cur_policy_shift_labels[cur_policy_shift_labels == self.tokenizer.pad_token_id] = -100
                            cur_policy_shift_labels = cur_policy_shift_labels.to(cur_policy_shift_labels.device)
                            cur_policy_reward_base_loss = loss_fct(
                                cur_policy_shift_logits, cur_policy_shift_labels.to(cur_policy_shift_logits.device)
                            ).reshape(logits.shape[0], -1)
                            original_dqn_reward = cur_policy_reward_base_loss.detach() - unreduced_loss
                                
                        if not did_skip_sampling:
                            nonzero_indices = prev_probabilities_2d.nonzero()
                            action_loglikelihoods = F.log_softmax(prev_sample_probs / self.reinforce_temperature, dim=-1)[nonzero_indices[:, 0], nonzero_indices[:, 1]]
                            action_loglikelihoods_2d = action_loglikelihoods.reshape(batch_size, -1)[:, :-1 - shift_amount]
                            action_loglikelihoods_list.append(action_loglikelihoods_2d)
                        if policy_reward is None:
                            policy_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                        else:
                            if self.n_ahead_talk > shift_amount:
                                added_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                            else:
                                added_reward = original_dqn_reward
                            policy_reward += added_reward
                    
                    if self.use_policy_loss and ahead_idx == self.n_ahead + self.n_ahead_talk - 2:
                        # only compute during the thinking phase
                        if self.use_reparam_for_thought_embeddings and (self.use_start_thought_token or self.use_end_thought_token):
                            # sampled_start, sampled_end
                            # calculate the log likelihood of the start and end embeddings sampled from a multivariate normal distribution
                            # with mean start_embedding[0] and standard deviation start_embedding[1]
                            if self.use_start_thought_token:
                                exp_start_std = torch.exp(start_embedding[1])
                                start_loglikelihood = -0.5 * (sampled_start.detach() - start_embedding[0]) ** 2 / exp_start_std ** 2 - start_embedding[1] - 0.5 * math.log(2 * math.pi)
                                start_loglikelihood = start_loglikelihood.mean(dim=-1)
                            if self.use_end_thought_token:
                                exp_end_std = torch.exp(end_embedding[1])
                                end_loglikelihood = -0.5 * (sampled_end.detach() - end_embedding[0]) ** 2 / exp_end_std ** 2 - end_embedding[1] - 0.5 * math.log(2 * math.pi)
                                end_loglikelihood = end_loglikelihood.mean(dim=-1)
                            # we use the mean instead of the sum to prevent dependence on the dimensionality of the embeddings
                            if self.use_end_thought_token and self.use_policy_loss_for_end_thought:
                                action_loglikelihoods_list.append(end_loglikelihood)
                            if self.use_start_thought_token:
                                action_loglikelihoods_list.append(start_loglikelihood)                                

                        if ahead_idx == self.n_ahead + self.n_ahead_talk - 2 and self.eval_mode:
                            with torch.no_grad():
                                # calculate the 0.75 quantile of the rewards
                                filtered_tokens = inputs_embeds[:, :policy_reward.shape[-1]].cpu().detach().numpy().flatten()
                                filtered_tokens_mask = filtered_tokens != self.tokenizer.pad_token_id
                                filtered_tokens = filtered_tokens[filtered_tokens_mask]
                                filtered_rewards = policy_reward.float().cpu().detach().numpy()[:, :seq_len - self.n_ahead_talk].flatten()
                                filtered_rewards = filtered_rewards[filtered_tokens_mask]

                                abs_reward_list = np.abs(policy_reward.float().cpu().detach().numpy()[:, :seq_len - self.n_ahead_talk].flatten())
                                abs_reward_list = abs_reward_list[filtered_tokens_mask]
                                medium_quantile = np.quantile(abs_reward_list, 0.5)
                                upper_quantile = np.quantile(abs_reward_list, 0.95)

                                save_tokens_with_rewards_to_pdf(
                                    filtered_tokens,
                                    [0] + filtered_rewards.tolist(),
                                    self.tokenizer,
                                    output_file=f"texts/rewards_talk_{self.n_ahead_talk}_{self.training_steps}.pdf",
                                    eps=medium_quantile,
                                    eps2=upper_quantile,
                                )

                                # def plot_kde(data, losses):
                                #     sns.set(style="whitegrid")
                                #     # Create the KDE plot
                                #     sns.kdeplot(data, fill=True)
                                #     # Set the plot title and labels
                                #     plt.title("KDE Plot")
                                #     plt.xlabel("Value")
                                #     plt.ylabel("Density")
                                #     # Save the plot
                                #     plt.savefig(f"texts/kde_talk_{self.n_ahead_talk}_{self.training_steps}.pdf")
                                #     # Close the plot
                                #     plt.close()

                                #     # Step 1: Create a base color palette
                                #     base_colors = sns.color_palette("light:#5A9", n_colors=256)  # More colors for a smoother gradient
                                #     base_cmap = LinearSegmentedColormap.from_list("log_light", base_colors)
                                #     log_norm = LogNorm(vmin=1e-3, vmax=10)

                                #     sns.kdeplot(x=data, y=losses, fill=True, levels=20, norm=log_norm, cut=0, linewidths=0)
                                #     # limit y to 0 to 25 and x to -1 to 1
                                #     plt.xlim(-1, 1)
                                #     plt.ylim(0, 25)
                                #     plt.savefig(f"texts/jointer_talk_{self.n_ahead_talk}_{self.training_steps}.pdf")
                                #     plt.close()

                                # self.all_rewards.extend(filtered_rewards)
                                # self.all_unreduced_losses.extend(unreduced_loss[:, :-1].flatten()[filtered_tokens_mask].float().flatten().cpu().detach().numpy())
                                # plot_kde(self.all_rewards, self.all_unreduced_losses)

                        for action_loglikelihoods_2d in action_loglikelihoods_list:
                            train_policy_reward = policy_reward

                            # discard rewards below the mean
                            if self.trice_mode and self.n_passes > 1:
                                batched_policy_reward = train_policy_reward.reshape(-1, self.n_passes, train_policy_reward.shape[-1])
                                # average over the passes
                                train_policy_reward = batched_policy_reward - batched_policy_reward.mean(dim=1, keepdim=True)
                                train_policy_reward = train_policy_reward.reshape(-1, train_policy_reward.shape[-1])
                                
                            if self.subtract_mean_reward:
                                train_policy_reward = train_policy_reward - train_policy_reward.mean()
                            if self.remove_negative_rewards:
                                fixed_policy_reward = train_policy_reward.detach().clamp(min=0)
                            else:
                                fixed_policy_reward = train_policy_reward.detach()
                            actor_loss = -fixed_policy_reward * action_loglikelihoods_2d[:, :policy_reward.shape[-1]].to(policy_reward.device)
                            if action_loglikelihoods_2d.mean() < -1e4 and not self.use_policy_loss_just_for_thoughts:
                                # This will only happen when we force the next token to be the end of thought token
                                break
                            dqn_loss_list.append(actor_loss.mean())

        if loss_list:
            if self.first_and_last_mode:
                loss = sum(
                    self.loss_mean(loss_list[-(i + 1)]) for i in range(self.n_ahead_talk)
                ) * (1 - self.original_loss_weight) / self.n_ahead_talk
                loss = loss + self.loss_mean(loss_list[0]) * self.original_loss_weight
                # Let's NaN out the others
                # e.g. if n_ahead_talk = 2 and the list is 5 long, we want to NaN out 1, 2 but keep 0, 3, 4
                for i in range(1, len(loss_list) - self.n_ahead_talk):
                    loss_list[i] = loss_list[i] * math.nan
            elif self.first_only:
                loss = self.loss_mean(loss_list[0])
            elif self.final_only_mode:
                loss = sum(
                    self.loss_mean(loss_list[-i]) for i in range(1, self.n_ahead_talk + 1)
                ) / self.n_ahead_talk   
            else:
                loss = None
                for i in range(len(loss_list)):
                    cur_loss = self.loss_mean(loss_list[i])
                    if loss is not None:
                        loss = loss + cur_loss.to(loss.device)
                    else:
                        loss = cur_loss
                loss = loss / len(loss_list)
            
            loss = loss * self.base_loss_beta

        if dqn_loss_list:
            dqn_loss = sum(dqn_loss_list) / len(dqn_loss_list)
            if self.include_policy_loss:
                if loss is not None:
                    loss += dqn_loss * self.policy_loss_beta
                else:
                    loss = dqn_loss * self.policy_loss_beta

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
    
        base_log_dict = {
            f"loss_{i}": nonzero_mean(loss_list[i]) for i in range(len(loss_list))
        }

        if loss is not None:
            base_log_dict["loss_train"] = loss.item()
        
        for loss_key, loss_val in base_log_dict.items():
            log_dict[loss_key] += loss_val / self.n_tokens_print
                
        if self.use_policy_loss and policy_reward is not None:
            log_dict["policy_loss"] += dqn_loss / self.n_tokens_print
            log_dict["policy_reward"] += policy_reward.mean() / self.n_tokens_print

        if not loss_list:
            if loss is not None:
                log_dict["loss_0"] += loss / self.n_tokens_print
        else:
            log_dict["loss_final"] += nonzero_mean(loss_list[-1]) / self.n_tokens_print
            log_dict["loss_talk"] += sum(nonzero_mean(cur_loss_item) for cur_loss_item in loss_list[-self.n_ahead_talk:]) / self.n_ahead_talk / self.n_tokens_print

        # also log relative losses to loss_0
        if loss_list:
            for i in range(len(loss_list)):
                talk_idx = min(max(i - (self.n_ahead - 1), 0), len(talk_loss_list) - 1)
                if not talk_loss_list:
                    cur_talk_loss = nonzero_mean(loss_list[0])
                else:
                    cur_talk_loss = talk_loss_list[talk_idx]
                log_dict[f"rel_loss_{i}"] += (nonzero_mean(loss_list[i]) - cur_talk_loss) / self.n_tokens_print
        if self.training:
            self.training_steps += 1
        try:
            # if self.training_steps % (self.gradient_accumulation_steps * 256) == 0:
            if self.wandb_enabled:
                if self.training_steps % (self.n_tokens_print) == 0 or not self.training:# and "0" in str(loss.device):
                    if not self.training:
                        new_log_dict = {}
                        for key in list(log_dict.keys()):
                            new_log_dict["eval_" + key] = log_dict[key]
                        log_dict = new_log_dict
                    log_dict["training_steps"] = self.training_steps 
                    log_dict["batch_size"] = batch_size
                    log_dict["example_steps"] = self.training_steps * batch_size * self.gradient_accumulation_steps
                    if self.n_ahead > 1:
                        log_dict["compute_steps"] = self.training_steps * batch_size * (self.n_ahead + self.n_ahead_talk - 1) * self.gradient_accumulation_steps
                    else: # There's no overhead for talk tokens if there's no thinking
                        log_dict["compute_steps"] = self.training_steps * batch_size * self.gradient_accumulation_steps
                    # remove all nans
                    for key in list(log_dict.keys()):
                        if log_dict[key] != log_dict[key]:
                            del log_dict[key]
                    if self.training:
                        wandb.log(log_dict)
                    if self.training:
                        self.log_dict = defaultdict(int)
                    else:
                        self.eval_log_dict = defaultdict(int)
        except Exception as e:
            pass

        if not self.training:
            self.n_ahead_talk = n_ahead_talk_to_restore
            self.n_passes = n_passes_to_restore
        return CausalLMOutputWithPast(
            loss=loss if loss is not None else None,
            logits=(rm_logits if self.n_ahead > 1 else logits) if not self.output_logits_at_the_end else logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
        
        
        
        
        # # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids = position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

        # return VolCanoCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     regression_loss=regression_loss,
        #     text_loss=text_loss
        # )

    def compute_snr(self,timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_caption(self, caption, length, inference=False):
        # add_special_tokens = False
        # if len(caption) == 0:
        add_special_tokens = True
        text_inputs = self.vision_generator.sd_tokenizer(
                caption,
                padding="max_length",
                max_length=length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=add_special_tokens
            ).to(self.device)
        # text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        prompt_embeds = self.vision_generator.sd_text_encoder(**text_inputs)[0]
        return prompt_embeds

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        original_input_ids = input_ids.clone()
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "input_images": kwargs.get("input_images", None),
                "box": kwargs.get("box", None),
                "image_label_masks": kwargs.get("image_label_masks", None),
            }
        )

        if 'input_ids' in model_inputs:
            new_token_ids = model_inputs['input_ids'][:, -1:]
            if new_token_ids == self.boi_token:
                #Generated the image token, force add all the image tokens
                next_inputs_embeds, current_target_image_embeds = self.generate_image(model_inputs)
                self.to_generate_images.append(current_target_image_embeds)
                model_inputs['input_ids'] = None
                model_inputs['inputs_embeds'] = next_inputs_embeds
                all_img_tokens_mask = torch.ones(1, self.n_query).to(device=model_inputs['attention_mask'].device, dtype=model_inputs['attention_mask'].dtype)
                model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], all_img_tokens_mask], dim=1)
            if new_token_ids == self.eoc_token_id and not self.no_bind:
                # need bounding box detection and use box align
                if self.sub_image_bind:
                    next_inputs_embeds, query_len = self.generate_sub_image(model_inputs, original_input_ids)
                else:
                    next_inputs_embeds, query_len = self.generate_box(model_inputs, original_input_ids)
                model_inputs['input_ids'] = None
                model_inputs['inputs_embeds'] = next_inputs_embeds
                all_img_tokens_mask = torch.ones(1, query_len).to(device=model_inputs['attention_mask'].device, dtype=model_inputs['attention_mask'].dtype)
                model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], all_img_tokens_mask], dim=1)
        return model_inputs

    def generate_box(self, model_inputs, original_input_ids):
        assert original_input_ids.shape[0] == 1
        if self.cache_images is None:
            self.cache_images = self.encode_img(model_inputs['input_images'][0][:1])[0]
        valid_start_ind = torch.where(original_input_ids==self.boc_token_id)[1].tolist()[-1]
        current_box_text = self.tokenizer.decode(original_input_ids[0, valid_start_ind:])
        current_box = torch.tensor(extract_box_str(current_box_text, mistral=True), dtype=self.dtype, device=self.cache_images.device)
        if current_box is None:
            print('fail to detect correct box from {}'.format(current_box_text))
            raise ValueError
        box_feat = self.box_align(self.cache_images[0], current_box.unsqueeze(0))[0]
        init_inputs_embeds = self.get_input_embeddings()(model_inputs['input_ids'])
        next_inputs_embeds = torch.cat([init_inputs_embeds, box_feat.unsqueeze(0)], dim=1)
        return next_inputs_embeds, box_feat.shape[0]

    def generate_sub_image(self, model_inputs, original_input_ids):
        assert original_input_ids.shape[0] == 1
        assert self.cache_raw_image is not None
        valid_start_ind = torch.where(original_input_ids==self.boc_token_id)[1].tolist()[-1]
        current_box_text = self.tokenizer.decode(original_input_ids[0, valid_start_ind:])
        current_box = extract_box_str(current_box_text, mistral=True)
        if current_box is None:
            print('fail to detect correct box from {}'.format(current_box_text))
            raise ValueError
        # box_feat = self.box_align(self.cache_images[0], current_box.unsqueeze(0))[0]
        w, h = self.cache_raw_image.size
        x_min, y_min, x_max, y_max = current_box
        x_min = x_min*w
        y_min = y_min*h
        x_max = x_max*w
        y_max = y_max*h
        sub_image = self.cache_raw_image.crop((x_min, y_min, x_max, y_max))
        sub_image = self.image_processor(resize_image_to_square(sub_image)).unsqueeze(0).to(dtype=self.dtype, device=original_input_ids.device)
        box_feat = self.encode_img(sub_image)[0][0]
        init_inputs_embeds = self.get_input_embeddings()(model_inputs['input_ids'])
        next_inputs_embeds = torch.cat([init_inputs_embeds, box_feat.unsqueeze(0)], dim=1)
        return next_inputs_embeds, box_feat.shape[0]
    
    def generate_image(self, model_inputs):
        input_ids = model_inputs['input_ids']
        past_key_values = model_inputs['past_key_values']
        use_cache = model_inputs['use_cache']
        attention_mask = model_inputs['attention_mask']
        bs = input_ids.shape[0]
        target_image_embeds = None
        init_inputs_embeds = self.get_input_embeddings()(input_ids)
        for num_img_token in range(self.n_query):
            if num_img_token == 0:
                inputs_embeds = init_inputs_embeds
            else:
                inputs_embeds = torch.cat([init_inputs_embeds, self.front_mm_projector(target_image_embeds)], dim=1)
            if past_key_values is None:
                target_shape = num_img_token + 1
            else:
                target_shape = past_key_values[-1][-1].shape[-2] + num_img_token + 1
            attention_mask = torch.cat((attention_mask, torch.ones(
                (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )), dim=1)
            sentence_length = torch.sum(attention_mask, dim=1).item()
            position_ids = torch.arange(sentence_length-(num_img_token+1), sentence_length).unsqueeze(0).to(attention_mask.device)
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[:, -(num_img_token+1):, :]
            target_image_embeds = self.behind_projector(target_image_embeds)
        inputs_embeds = torch.cat([init_inputs_embeds, self.front_mm_projector(target_image_embeds)], dim=1)
        return inputs_embeds, target_image_embeds
    
    def flatten_hidden(self, hidden_state_list):
        last_hidden_state = []
        for hid_st in hidden_state_list:
            last_hidden_state.append(hid_st[-1])
        last_hidden_state = torch.cat(last_hidden_state,dim=1)
        return last_hidden_state

    def condition_completion(self, input_dict, temperature=0.2, max_new_tokens=128, guidance_scale=7.5, avoid_image_gen=False, **kwargs):
        
        self.to_generate_images = []
        self.cache_images = None
        self.boi_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_BOI_TOKEN])[0]
        if self.sub_image_bind:
            self.cache_raw_image = input_dict['raw_images'][0][0]
        else:
            self.cache_raw_image = None
        del(input_dict['raw_images'])
        if len(input_dict['input_ids'].shape) == 1:
            input_dict['input_ids'] = input_dict['input_ids'].unsqueeze(0)
            input_dict['attention_mask'] = input_dict['attention_mask'].unsqueeze(0)
        if isinstance(input_dict['input_images'], list):
            input_dict['input_images'] = [item.to(self.dtype).to(self.device) if item is not None else item for item in input_dict['input_images']]
        else:
            input_dict['input_images'] = [input_dict['input_images'].to(self.dtype).to(self.device)] if input_dict['input_images'] is not None else [None]
        with torch.no_grad():
            text_out = self.generate(
                        input_ids = input_dict['input_ids'].to(self.device), # TODO unsqueeze is for bs==1
                        input_images=input_dict['input_images'] if input_dict['input_images'] is not None else [None],
                        attention_mask = input_dict['attention_mask'].to(self.device),
                        box = input_dict['box'] if 'box' in input_dict else None,
                        do_sample = True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens = max_new_tokens,
                        pad_token_id = self.tokenizer.pad_token_id,
                        return_dict_in_generate = True
                    )

        input_token_len = input_dict['input_ids'].shape[1]
        n_diff_input_output = (input_dict['input_ids'].to(self.device) != text_out.sequences[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        pred_out = self.tokenizer.batch_decode(text_out.sequences[:, input_token_len:], skip_special_tokens=False)
        
        # check if images are in the output, if so, decode them
        output_images = []
        if len(self.to_generate_images) and not avoid_image_gen:
            self.vision_generator.image_pipeline.to(self.device, self.dtype)
            negative_embeds = self.encode_img(torch.zeros(1, 3, self.vision_encoder.image_size, self.vision_encoder.image_size).to(device=self.device, dtype=self.dtype))[2]
            for to_gen_img in self.to_generate_images:
                out_img = self.vision_generator.image_pipeline(prompt_embeds=to_gen_img, negative_embeds=negative_embeds, guidance_scale=3, height=1024, width=1024, crop_info=[0, 0], original_size=[1024, 1024], num_inference_steps=100).image         
                output_images.append(out_img)
        return pred_out, output_images, text_out.sequences

    def calculate_options(self, input_dict, cot=False, further_instruct=False, temperature=0.2, max_new_tokens=128, guidance_scale=7.5, avoid_image_gen=False, likelihood_reduction='mean', **kwargs):
        
        assert len(input_dict['options']) == 1
        self.cache_images = None
        self.boi_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_BOI_TOKEN])[0]
        options = [s for s in input_dict['options'][0]]
        del(input_dict['options'])
        if self.sub_image_bind:
            self.cache_raw_image = input_dict['raw_images'][0][0]
        else:
            self.cache_raw_image = None
        del(input_dict['raw_images'])
        if len(input_dict['input_ids'].shape) == 1:
            input_dict['input_ids'] = input_dict['input_ids'].unsqueeze(0)
            input_dict['attention_mask'] = input_dict['attention_mask'].unsqueeze(0)
        if isinstance(input_dict['input_images'], list):
            input_dict['input_images'] = [item.to(self.dtype).to(self.device) if item is not None else item for item in input_dict['input_images']]
        else:
            input_dict['input_images'] = [input_dict['input_images'].to(self.dtype).to(self.device)] if input_dict['input_images'] is not None else [None]
        if cot:
            # need to first conduct the thinking
            with torch.no_grad():
                text_out = self.generate(
                            input_ids = input_dict['input_ids'].to(self.device), # TODO unsqueeze is for bs==1
                            input_images=input_dict['input_images'] if input_dict['input_images'] is not None else [None],
                            attention_mask = input_dict['attention_mask'].to(self.device),
                            box = input_dict['box'] if 'box' in input_dict else None,
                            do_sample = True if temperature > 0 else False,
                            temperature=temperature,
                            max_new_tokens = max_new_tokens,
                            pad_token_id = self.tokenizer.pad_token_id,
                            return_dict_in_generate = True
                        )
            thought_ids = text_out.sequences
            input_token_len = input_dict['input_ids'].shape[1]
            n_diff_input_output = (input_dict['input_ids'].to(self.device) != text_out.sequences[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            thought = self.tokenizer.batch_decode(text_out.sequences[:, input_token_len:], skip_special_tokens=False)[0]
            thought_ids = thought_ids.squeeze()
            if further_instruct:
                # need to instruct the model to select from options!
                options_instruct = 'Select from following options: ' + '; '.join(options) + '.'
                option_instruct_ids = self.tokenizer([options_instruct], return_tensors='pt')['input_ids'][:, 1:].squeeze()
                suffix = torch.cat([torch.tensor([733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804]), 
                                    option_instruct_ids, 
                                    torch.tensor([733, 28748, 16289, 28793])]).to(thought_ids.device)
            else:
                suffix = torch.tensor([  733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                                        28748, 16289, 28793]).to(thought_ids.device)
            eoc_indices = [-1] + torch.where(thought_ids == self.eoc_token_id)[0].tolist() + [thought_ids.shape[0]-1]
            input_ids = []
            for i in range(len(eoc_indices) - 1):
                input_ids.append(thought_ids[eoc_indices[i]+1:eoc_indices[i+1]+1])
                if i < len(eoc_indices) - 2:
                    if thought_ids[eoc_indices[i+1]+1].item() != self.input_img_id:
                        input_ids.append(torch.tensor([self.input_img_id]).to(thought_ids.device))
            input_ids.append(suffix)
            input_ids = torch.cat(input_ids).unsqueeze(0)
            new_thought, thought_boxes = process_thought(thought, mistral=True)
            if self.sub_image_bind:
                w, h = self.cache_raw_image.size
                new_sub_images = []
                for b in thought_boxes:
                    x_min, y_min, x_max, y_max = b
                    x_min = x_min*w
                    y_min = y_min*h
                    x_max = x_max*w
                    y_max = y_max*h
                    sub_image = self.cache_raw_image.crop((x_min, y_min, x_max, y_max))
                    new_sub_images.append(self.image_processor(resize_image_to_square(sub_image)).unsqueeze(0).to(dtype=self.dtype, device=input_ids.device))
                num_images = len(new_sub_images) + 1
                input_dict['input_images'] = [torch.cat(input_dict['input_images'] + new_sub_images, dim=0)]
                all_box = [torch.tensor([[0.0, 0.0, 1.0, 1.0]]*num_images, device=input_ids.device, dtype=input_dict['box'][0].dtype)]
            else: 
                all_box = input_dict['box'][0]
                all_box = [torch.cat([all_box, torch.tensor(thought_boxes, device=all_box.device, dtype=all_box.dtype)], dim=0)]
        else:
            input_ids = input_dict['input_ids']
            all_box = input_dict['box']
        
        # calculate the past qk for processing
        input_dict['input_ids'] = input_ids.to(self.device)
        input_dict['box'] = all_box
        input_dict['use_cache'] = True
        input_dict['attention_mask'] = torch.ones_like(input_dict['input_ids']).to(self.device)
        del(input_dict['labels'])
        question_output = self.forward(**input_dict)
        question_logits = question_output.logits

        # calculate the logit
        option_losses = []
        for opt in options:
            opt_ids = self.tokenizer([opt], return_tensors='pt')['input_ids'][:, 1:].to(self.device)
            opt_output = self.forward(input_ids = opt_ids, attention_mask=torch.ones_like(opt_ids), 
                                    past_key_values=question_output.past_key_values, use_cache=True)
            logits = torch.cat([question_logits[:, -1:], opt_output.logits[:, :-1]], 1)
            if likelihood_reduction == 'mean':
                loss_fct = CrossEntropyLoss()
                logits = logits.view(-1, self.config.vocab_size)
                labels = opt_ids.view(-1)
                loss = loss_fct(logits, labels)
            elif likelihood_reduction == 'sum':
                loss_fct = CrossEntropyLoss(reduction='none')
                logits = logits.view(-1, self.config.vocab_size)
                labels = opt_ids.view(-1)
                loss = loss_fct(logits, labels).sum()
            else:
                raise ValueError
            option_losses.append(loss)
        
        return torch.stack(option_losses).argmin().cpu().item(), thought if cot else None

    def load_state_dict_from_ckpt(self,ckpt_file):
        state_dict = torch.load(ckpt_file)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = dict()
        for key in state_dict.keys():
            if 't2i_decoder_prompt' in key or 'llm_to_t2i_mapping' in key:
                new_key = 'behind_projector.'+key
            elif 'llama_model' in key:
                if 'lora' in key or 'modules_to_save' in key:
                    new_key = '.'.join(key.split('.')[4:])
                else:
                    new_key = '.'.join(key.split('.')[2:])
            new_state_dict[new_key] = state_dict[key]

        model_state_dict = self.state_dict()
        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)

    def load_state_dict_from_old_code_checkpoint(self,ckpt_file):
        state_dict = torch.load(ckpt_file)
        prefix_key = list(set(['.'.join(key.split('.')[:2]) for key in state_dict.keys()]))
        new_state_dict = dict()
        for key in state_dict.keys():
            if 't2i_decoder_prompt' in key or 'llm_to_t2i_mapping' in key:
                new_key = 'behind_projector.'+key
            elif 'llama_model' in key:
                new_key = '.'.join(key.split('.')[4:])
            elif 'vae' in key or 'unet' in key or 'sd_text_encoder' in key:
                new_key = 'vision_generator.'+key
                # check the unet condition conv_in layer
                if 'unet.conv_in.weight' == key:
                    num_channels = state_dict[key].shape[1]
                    if self.image_condition:
                        if self.config.vision_generator_cond_channels + 4 != num_channels:
                            continue
            elif 'Qformer' in key:
                new_key = 'front_mm_projector.'+'.'.join(key.split('.')[2:])
            elif 'query_tokens' in key or 'llama_proj' in key:
                new_key = '.'.join(key.split('.')[1:])
            elif 'ln_vision' in key:
                new_key = 'vit_ln.' + '.'.join(key.split('.')[1:])
            elif 'visual_encoder' in key:
                new_key = 'vision_encoder.'+ '.'.join(key.split('.')[2:])
            elif 'fc' in key:
                new_key = key
            else:
                raise ValueError('no support key from old code checkpoint')

            new_state_dict[new_key] = state_dict[key]

        self.load_state_dict(new_state_dict, strict=False)

AutoConfig.register("VolCanoMistral", VolCanoMistralConfig)
AutoModelForCausalLM.register(VolCanoMistralConfig, QuietVolCanoMistralForCausalLM)
