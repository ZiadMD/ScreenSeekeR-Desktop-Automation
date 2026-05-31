"""
GUI-Actor model adapter for ScreenSeekeR integration.

Handles model loading, inference, and conversion of attention-based
grounding output into normalized coordinate format compatible with the
existing Grounder pipeline.

Adapted from microsoft/GUI-Actor inference code (MIT License).
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from src.utils.logging import logger

# Special token strings used by GUI-Actor
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

# System prompt for GUI-Actor grounding
GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
    "your task is to locate the screen element that corresponds to the instruction. "
    "You should output a PyAutoGUI action that performs a click on the correct position. "
    "To indicate the click location, we will use some special tokens, which is used to "
    "refer to a visual patch later. For example, you can output: "
    "pyautogui.click(<your_special_token_here>)."
)

# Chat template for Qwen2.5-VL (from GUI-Actor constants)
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


class GUIActorAdapter:
    """
    Loads and runs inference on GUI-Actor-3B-Qwen2.5-VL for GUI element grounding.
    
    Converts the attention-based output (action head scores over visual patches)
    into normalized (0-1) coordinates compatible with the existing ScreenSeekeR pipeline.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        attn_impl: str = "sdpa",
        max_pixels: int = 3200 * 1800,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.attn_impl = attn_impl
        self.max_pixels = max_pixels

        # Lazy-loaded on first inference call
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._logits_processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        """Load the GUI-Actor model and processor from local weights."""
        import torch
        from transformers import AutoProcessor, LogitsProcessorList

        logger.info(f"Loading GUI-Actor model from: {self.model_path}")
        logger.info(f"Device: {self.device}, dtype: {self.torch_dtype_str}, attn: {self.attn_impl}")

        # Resolve torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float16)

        # Import the custom model class
        from src.grounding.local_model.modeling_qwen25vl import (
            Qwen2_5_VLForConditionalGenerationWithPointer
        )

        # Load processor (handles image + text preprocessing)
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._tokenizer = self._processor.tokenizer

        # Load model with the custom action head
        self._model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device,
            attn_implementation=self.attn_impl,
        ).eval()

        # Create the logits processor for forcing pointer token sequences
        from src.grounding.local_model._inference_utils import ForceFollowTokensLogitsProcessor
        pointer_start_id = self._tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        pointer_pad_id = self._tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        pointer_end_id = self._tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]

        self._logits_processor = ForceFollowTokensLogitsProcessor(
            token_a_id=pointer_start_id,
            forced_sequence=[pointer_pad_id, pointer_end_id]
        )

        logger.info(f"GUI-Actor model loaded successfully. "
                     f"Pointer tokens: start={pointer_start_id}, pad={pointer_pad_id}, end={pointer_end_id}")

    def _ensure_loaded(self):
        """Ensure model is loaded (lazy initialization)."""
        if not self.is_loaded:
            self._load_model()

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to fit within max_pixels while maintaining aspect ratio."""
        w, h = image.size
        current_pixels = w * h
        if current_pixels > self.max_pixels:
            ratio = (self.max_pixels / current_pixels) ** 0.5
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            image = image.resize((new_w, new_h))
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        return image

    def ground_element(self, image: Image.Image, instruction: str) -> Dict[str, Any]:
        """
        Run GUI-Actor inference to ground a UI element in the image.

        Args:
            image: Screenshot or crop to search in.
            instruction: Natural language description of the element to find.

        Returns:
            Dict matching existing Grounder output format:
            {
                "x": float,        # Normalized center X (0-1)
                "y": float,        # Normalized center Y (0-1)
                "width": float,    # Estimated bbox width (0-1)
                "height": float,   # Estimated bbox height (0-1)
                "confidence": float,
                "reasoning": str
            }
        """
        import torch
        from transformers import LogitsProcessorList
        from qwen_vl_utils import process_vision_info

        self._ensure_loaded()

        # Resize image if needed
        image = self._resize_image(image.convert("RGB"))
        img_w, img_h = image.size

        # Build conversation in GUI-Actor format
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": GROUNDING_SYSTEM_MESSAGE}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=CHAT_TEMPLATE
        )

        # Prepare inputs
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self._model.device)

        # Remove keys the model's generate() doesn't accept
        # (newer transformers processors may produce mm_token_type_ids)
        inputs.pop("mm_token_type_ids", None)

        # Reset the logits processor state for this inference
        self._logits_processor.force_queue = []

        # Run generation
        with torch.inference_mode():
            results = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                logits_processor=LogitsProcessorList([self._logits_processor]),
                return_dict_in_generate=True,
                output_hidden_states=True
            )

        # Decode output text
        input_ids = inputs["input_ids"][0]
        generated_ids = results.sequences[0][len(input_ids):]
        output_text = self._tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        logger.debug(f"GUI-Actor generated: {output_text[:200]}")

        # Check for pointer tokens in generated output
        pointer_pad_mask = (generated_ids[:-1] == self._model.config.pointer_pad_token_id)

        if pointer_pad_mask.sum() == 0:
            logger.warning("GUI-Actor did not generate any pointer tokens. Falling back to center.")
            return {
                "x": 0.5, "y": 0.5,
                "width": 0.2, "height": 0.2,
                "confidence": 0.1,
                "reasoning": "No pointer tokens generated by GUI-Actor"
            }

        # Extract hidden states for pointer tokens
        decoder_hidden_states = [
            step_hidden_states[-1][0]
            for step_hidden_states in results.hidden_states[1:]
        ]
        decoder_hidden_states = torch.cat(decoder_hidden_states, dim=0)
        decoder_hidden_states = decoder_hidden_states[pointer_pad_mask]

        # Get image embeddings from the first generation step
        image_pad_token_id = self._tokenizer.encode("<|image_pad|>")[0]
        image_mask = (inputs["input_ids"][0] == image_pad_token_id)
        image_embeds = results.hidden_states[0][0][0][image_mask]

        # Run the action head (pointer network)
        attn_scores, _ = self._model.multi_patch_pointer_head(image_embeds, decoder_hidden_states)

        # Get patch grid dimensions
        _, n_height, n_width = (
            inputs["image_grid_thw"][0] // self._model.visual.spatial_merge_size
        ).tolist()

        # Convert attention scores to coordinates
        best_point, region_centers, region_scores, _ = _get_prediction_region_point(
            attn_scores, n_width, n_height,
            return_all_regions=True, rect_center=False
        )

        # best_point is (x, y) in normalized (0-1) coordinates
        pred_x, pred_y = best_point
        confidence = region_scores[0] if region_scores else 0.5

        # Estimate bounding box from the top region's patch spread
        width_est, height_est = _estimate_bbox_from_patches(
            attn_scores, n_width, n_height
        )

        logger.info(f"GUI-Actor grounded at ({pred_x:.3f}, {pred_y:.3f}) "
                     f"confidence={confidence:.3f}, bbox≈{width_est:.3f}x{height_est:.3f}")

        return {
            "x": float(pred_x),
            "y": float(pred_y),
            "width": float(width_est),
            "height": float(height_est),
            "confidence": float(min(confidence * 2.0, 1.0)),  # Scale up since attention scores are typically low
            "reasoning": f"GUI-Actor attention grounding: {output_text[:100]}"
        }


def _get_prediction_region_point(
    attn_scores, n_width, n_height,
    activation_threshold=0.3,
    return_all_regions=True,
    rect_center=False
):
    """
    Extract the best prediction point from attention scores over a patch grid.
    
    Adapted from GUI-Actor's get_prediction_region_point().
    
    1. Select activated patches (above threshold * max)
    2. Cluster connected patches into regions via BFS
    3. Rank regions by average activation
    4. Return weighted center of best region
    """
    import torch

    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]

    # Convert flat indices to 2D grid coordinates
    topk_coords = []
    for idx in valid_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

    # BFS to find connected regions
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue

        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]

        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))

        regions.append(region)

    if not regions:
        # Fallback: use the global max patch
        max_idx = attn_scores[0].argmax().item()
        center_x = ((max_idx % n_width) + 0.5) / n_width
        center_y = ((max_idx // n_width) + 0.5) / n_height
        return (center_x, center_y), [(center_x, center_y)], [max_score], [[(center_x, center_y)]]

    # Score and sort regions
    region_scores = []
    region_centers = []
    region_points = []

    for region in regions:
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)

        normalized_centers = []
        weights = []
        for y, x, _, score in region:
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

        region_points.append(normalized_centers)

        if not rect_center:
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            region_centers.append((weighted_x, weighted_y))
        else:
            x_coords = set(nc[0] for nc in normalized_centers)
            y_coords = set(nc[1] for nc in normalized_centers)
            region_centers.append((sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)))

    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]

    return sorted_centers[0], sorted_centers, sorted_scores, sorted_points


def _estimate_bbox_from_patches(attn_scores, n_width, n_height, threshold_ratio=0.3):
    """
    Estimate a bounding box size from the spread of activated patches.
    Returns (width, height) as normalized (0-1) fractions.
    """
    import torch

    max_score = attn_scores[0].max().item()
    threshold = max_score * threshold_ratio
    mask = attn_scores[0] > threshold
    indices = torch.nonzero(mask).squeeze(-1)

    if indices.numel() == 0:
        return 0.1, 0.1

    ys = indices // n_width
    xs = indices % n_width

    y_span = (ys.max().item() - ys.min().item() + 1) / n_height
    x_span = (xs.max().item() - xs.min().item() + 1) / n_width

    # Clamp to reasonable range
    return max(0.02, min(0.5, x_span)), max(0.02, min(0.5, y_span))
