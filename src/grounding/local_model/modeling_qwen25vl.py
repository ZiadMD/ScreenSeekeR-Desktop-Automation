"""
Vendored and adapted from microsoft/GUI-Actor (MIT License)
https://github.com/microsoft/GUI-Actor/blob/main/src/gui_actor/modeling_qwen25vl.py

Custom Qwen2.5-VL model class with an attention-based action head (pointer network)
for coordinate-free GUI grounding.

Modifications from original:
- Removed dependency on gui_actor.constants (inlined IGNORE_INDEX)
- Removed dependency on gui_actor.trainer (replaced rank0_print with logging)
- Made compatible with standalone import
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration
)
from typing import List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

# Inlined from gui_actor.constants
IGNORE_INDEX = -100


def rank0_print(*args, **kwargs):
    """Replacement for gui_actor.trainer.rank0_print — just logs."""
    logger.debug(" ".join(str(a) for a in args))


class QwenVLwithVisionHeadOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output class for Qwen2_5_VL with pointer head, extending the base output class.

    Args:
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        pointer_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vision pointer network loss.
        pointer_scores (`List[torch.FloatTensor]`, *optional*):
            Attention scores from the pointer network, one tensor per batch item.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (weighted sum of lm_loss and pointer_loss).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores from the language modeling head.
        past_key_values, hidden_states, attentions, rope_deltas:
            Same as parent class.
    """
    def __init__(self, lm_loss=None, pointer_loss=None, pointer_scores=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_loss = lm_loss
        self.pointer_loss = pointer_loss
        self.pointer_scores = pointer_scores


class VisionHead_MultiPatch(nn.Module):
    def __init__(self, d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model

        # Note: We omit additional normalization here because Qwen2VL
        # already normalizes hidden states using RMSNorm.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        # Add self-attention layer for visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                hidden_state_enc,  # shape: [n_enc, d_model] where n_enc can vary with image size
                hidden_state_dec,  # shape: [n_dec, d_model] there can be multiple query in one sample
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):

        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            need_weights=False
        )
        # Residual connection and layer normalization
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        # Remove batch dimension
        hidden_state_enc_ctx = hidden_state_enc_ctx.squeeze(0)  # [n_enc, d_model]

        # Apply the projection networks.
        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]

        # Compute scaled dot-product attention scores.
        # Scaling by sqrt(d_model) is critical regardless of variable n_enc.
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]

        # Softmax normalization is applied along the encoder dimension.
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        if (labels is not None) and (not do_single_patch):
            epsilon = 1e-8
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            # Apply log_softmax to logits
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

        if do_single_patch and (labels is not None):
            loss = F.cross_entropy(patch_logits, labels)

        return attn_weights, loss


class Qwen2_5_VLForConditionalGenerationWithPointer(Qwen2_5_VLForConditionalGeneration):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Override from_pretrained to remap checkpoint keys for transformers v5.x compatibility.
        
        The GUI-Actor checkpoint was saved with transformers v4.x which used flat key names:
            model.layers.*, model.embed_tokens.*, model.norm.*
        But transformers v5.x nests these under model.language_model:
            model.language_model.layers.*, model.language_model.embed_tokens.*, etc.
        """
        import os
        from safetensors.torch import load_file as safetensors_load

        # Check if we need to remap by looking at the checkpoint keys
        model_path = str(pretrained_model_name_or_path)
        needs_remap = False

        if os.path.isdir(model_path):
            # Find safetensors files
            shard_files = sorted([
                os.path.join(model_path, f)
                for f in os.listdir(model_path)
                if f.endswith('.safetensors') and not f.endswith('.index.json')
            ])
            if shard_files:
                # Quick check: peek at first file's keys
                first_shard = safetensors_load(shard_files[0])
                sample_key = next(iter(first_shard.keys()), "")
                needs_remap = sample_key.startswith("model.layers.") or sample_key.startswith("model.embed_tokens.")
                del first_shard

        if needs_remap:
            logger.info("Detected old-format checkpoint keys. Remapping for transformers v5.x compatibility...")

            # Load all shards, remap keys
            remapped_state_dict = {}
            for shard_path in shard_files:
                shard = safetensors_load(shard_path)
                for key, value in shard.items():
                    new_key = cls._remap_key(key)
                    remapped_state_dict[new_key] = value
                del shard

            logger.info(f"Remapped {len(remapped_state_dict)} keys from checkpoint.")

            # Load model architecture (weights will be random initially)
            # We pass ignore_mismatched_sizes to avoid errors during initial load
            original_init_weights = kwargs.pop('_fast_init', True)
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                ignore_mismatched_sizes=True,
                **kwargs
            )

            # Now load the correctly-remapped weights
            missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
            # Filter out expected missing keys (pointer head is newly initialized)
            real_missing = [k for k in missing if 'multi_patch_pointer_head' not in k]
            if real_missing:
                logger.warning(f"Still missing {len(real_missing)} keys after remap: {real_missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys after remap: {unexpected[:5]}...")
            logger.info("Remapped state dict loaded successfully.")

            del remapped_state_dict
            return model
        else:
            # Standard loading path — no remapping needed
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

    @staticmethod
    def _remap_key(key: str) -> str:
        """
        Remap a single checkpoint key from v4.x format to v5.x format.
        
        v4.x (checkpoint):                    v5.x (expected):
        model.layers.*                    →   model.language_model.layers.*
        model.embed_tokens.*              →   model.language_model.embed_tokens.*
        model.norm.*                      →   model.language_model.norm.*
        lm_head.*                         →   lm_head.* (unchanged)
        visual.*                          →   visual.* (unchanged, but may need model. prefix)
        multi_patch_pointer_head.*        →   multi_patch_pointer_head.* (unchanged)
        """
        # Remap text model keys from flat to nested language_model
        if key.startswith("model.layers."):
            return key.replace("model.layers.", "model.language_model.layers.", 1)
        elif key.startswith("model.embed_tokens."):
            return key.replace("model.embed_tokens.", "model.language_model.embed_tokens.", 1)
        elif key.startswith("model.norm."):
            return key.replace("model.norm.", "model.language_model.norm.", 1)
        # Remap vision encoder keys: visual.* → model.visual.*
        elif key.startswith("visual."):
            return "model." + key
        # Everything else stays the same
        return key

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compatibility shim for transformers v5.x+
        # In newer versions, Qwen2.5-VL config nests text model attributes
        # under config.text_config instead of exposing them at the top level.
        # The VisionHead and forward() code needs these at the top level.
        if not hasattr(self.config, 'hidden_size') and hasattr(self.config, 'text_config'):
            text_cfg = self.config.text_config
            for attr in ('hidden_size', 'vocab_size', 'num_attention_heads',
                         'num_hidden_layers', 'intermediate_size',
                         'image_token_id', 'video_token_id',
                         'pointer_start_token_id', 'pointer_pad_token_id',
                         'pointer_end_token_id'):
                if hasattr(text_cfg, attr) and not hasattr(self.config, attr):
                    setattr(self.config, attr, getattr(text_cfg, attr))
            logger.info("Applied text_config compatibility shim for transformers v5.x+")

        self.multi_patch_pointer_head = VisionHead_MultiPatch(self.config.hidden_size, self.config.hidden_size)
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        self.post_init()

    def reset_loss_weights(self, pointer_loss_weight, lm_loss_weight):
        self.pointer_loss_weight = pointer_loss_weight
        self.lm_loss_weight = lm_loss_weight

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                second_per_grid_ts: Optional[torch.Tensor] = None,
                # Grounding
                visual_token_indices_of_coordinates: Optional[torch.Tensor] = None,
                multi_patch_labels: Optional[torch.Tensor] = None,
                if_multi_patch: bool = True,
                coordinates: Optional[List[Tuple[float, float]]] = None,
                verbose: bool = False) -> Union[Tuple, QwenVLwithVisionHeadOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if verbose:
            rank0_print(f"input_ids: {input_ids.shape}, {input_ids[0][:5]}...")
            rank0_print(f"labels: {labels.shape}, {labels[0][:5]}...")
            rank0_print(f"pixel_values: {pixel_values.shape}")
            rank0_print(f"image_grid_thw: {image_grid_thw.shape}, {image_grid_thw}")
            rank0_print(f"coordinates: {coordinates}")
            rank0_print(f"visual_token_indices_of_coordinates: {visual_token_indices_of_coordinates}")
            rank0_print(f"return_dict: {return_dict}")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore.
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]  # shape: (batch_size, seq_len, d_model)
        logits = self.lm_head(hidden_states)

        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

        # If vision supervision is requested, process the action head.
        pointer_loss = None
        pointer_scores = []
        if visual_token_indices_of_coordinates is not None:
            batch_size = input_ids.shape[0]
            pointer_losses = []

            for i in range(batch_size):
                dummy_target = False
                token_ids = input_ids[i]
                hs = hidden_states[i]

                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)

                target_mask = (token_ids == self.config.pointer_pad_token_id)
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

                if visual_indices.numel() == 0:
                    raise ValueError(f"No visual or target tokens found for sample {i}.")
                if target_indices.numel() == 0:
                    target_indices = torch.tensor([hs.shape[0] - 1])
                    gt = torch.tensor([0]).to(hs.device)
                    if if_multi_patch:
                        sample_labels = torch.zeros_like(visual_indices).unsqueeze(0)
                        sample_labels[0][:4] = 1
                    dummy_target = True
                else:
                    gt = visual_token_indices_of_coordinates[i].to(hs.device)
                    if if_multi_patch:
                        sample_labels = multi_patch_labels[i]

                visual_embeds = inputs_embeds[i][visual_indices]
                target_hidden = hs[target_indices]

                if if_multi_patch:
                    if sample_labels.shape[0] != target_indices.shape[0]:
                        raise ValueError(f"Sample {i} has mismatched target counts: {sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens")
                    attn_scores, loss_v = self.multi_patch_pointer_head(
                        visual_embeds,
                        target_hidden,
                        labels=sample_labels
                    )
                else:
                    attn_scores, loss_v = self.pointer_head(visual_embeds, target_hidden, labels=gt)

                pointer_scores.append(attn_scores.detach().cpu())
                pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)

            pointer_loss = torch.stack(pointer_losses).mean()

        if lm_loss is None:
            total_loss = pointer_loss
        elif pointer_loss is None:
            total_loss = lm_loss
        else:
            total_loss = self.lm_loss_weight * lm_loss + self.pointer_loss_weight * pointer_loss

        if return_dict:
            return QwenVLwithVisionHeadOutputWithPast(
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                pointer_scores=pointer_scores,
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
        else:
            if labels is not None:
                output = (lm_loss, pointer_loss, logits, pointer_scores,) + outputs[1:]
                return (total_loss,) + output if total_loss is not None else output
            else:
                return outputs
