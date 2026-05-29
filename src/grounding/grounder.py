import json
import re
from typing import Tuple, Dict, Any, Optional
from PIL import Image
from src.grounding.llm_client import LLMClient
from src.utils.logging import logger

SYSTEM_PROMPT = """
You are a precise GUI grounding assistant. Your task is to locate a specific UI element within a cropped screenshot based on a natural language description.
You must analyze the image carefully and find the center point (x, y) and size (width, height) of the element matching the description.

All coordinates and sizes MUST be normalized between 0.0 and 1.0 relative to the image's dimensions.
For example, if the target is in the exact center of the cropped image, (x, y) would be (0.5, 0.5).

You MUST output your response strictly as a JSON object with these keys:
{
  "x": float,          // Center X coordinate (0.0 to 1.0)
  "y": float,          // Center Y coordinate (0.0 to 1.0)
  "width": float,      // Width of bounding box (0.0 to 1.0)
  "height": float,     // Height of bounding box (0.0 to 1.0)
  "confidence": float, // Confidence score (0.0 to 1.0)
  "reasoning": str     // Brief visual explanation of why this matches
}
Do NOT include any markdown code blocks, backticks, or other text outside of the JSON object.
"""

USER_PROMPT_TEMPLATE = "Find the precise location of the UI element described as: '{instruction}' within this crop."

class Grounder:
    """
    Grounder module responsible for high-precision element location within image crops.
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    def ground_element(self, crop: Image.Image, instruction: str) -> Dict[str, Any]:
        """
        Sends the crop and target instruction to the Vision MLLM.
        Returns a dictionary containing normalized coordinates, size, confidence, and reasoning.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(instruction=instruction)
        
        logger.info(f"Grounding instruction '{instruction}' in cropped region of size {crop.width}x{crop.height}...")
        
        try:
            response_text = self.client.call_vision_api(
                image=crop,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_response=True
            )
            
            parsed_json = self._parse_json_from_response(response_text)
            logger.info(f"Grounder response parsed successfully: confidence={parsed_json.get('confidence', 0.0):.1%}")
            logger.debug(f"Grounder result details: {parsed_json}")
            
            # Ensure coordinates are within bounds
            for key in ["x", "y", "width", "height", "confidence"]:
                if key in parsed_json:
                    parsed_json[key] = max(0.0, min(1.0, float(parsed_json[key])))
                    
            return parsed_json
            
        except Exception as e:
            logger.error(f"Failed to ground element inside crop: {e}")
            raise

    def _parse_json_from_response(self, text: str) -> Dict[str, Any]:
        """
        Robustly extracts and parses a JSON object from raw model response text,
        handling Markdown wrapper backticks or extra text if present.
        """
        text = text.strip()
        
        # Try to find JSON inside markdown code blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Or find the first open brace and last close brace
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        return json.loads(text)

def map_relative_to_absolute(
    rel_x: float, rel_y: float,
    rel_w: float, rel_h: float,
    crop_box: Tuple[int, int, int, int]
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
    """
    Maps relative coordinates (0-1) inside a crop back to absolute physical coordinates on the full screen.
    
    crop_box: (x_min, y_min, x_max, y_max) in physical pixels.
    Returns:
        - Bounding Box absolute: (x1, y1, x2, y2) in physical pixels
        - Center absolute: (cx, cy) in physical pixels
    """
    x_min, y_min, x_max, y_max = crop_box
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    
    # Calculate absolute center
    abs_cx = x_min + (rel_x * crop_w)
    abs_cy = y_min + (rel_y * crop_h)
    
    # Calculate absolute width & height
    abs_w = rel_w * crop_w
    abs_h = rel_h * crop_h
    
    # Calculate bounding box bounds
    abs_x1 = abs_cx - (abs_w / 2)
    abs_y1 = abs_cy - (abs_h / 2)
    abs_x2 = abs_cx + (abs_w / 2)
    abs_y2 = abs_cy + (abs_h / 2)
    
    return (abs_x1, abs_y1, abs_x2, abs_y2), (abs_cx, abs_cy)
