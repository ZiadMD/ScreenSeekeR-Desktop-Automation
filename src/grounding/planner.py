import json
import re
from typing import Dict, Any, List, Optional
from PIL import Image
from src.grounding.llm_client import LLMClient
from src.utils.logging import logger

SYSTEM_PROMPT = """
You are an expert GUI planner assistant.
Your task is to analyze a high-resolution desktop screenshot and identify the most likely regions (candidate bounding boxes) where a requested UI element is located.

Since UI elements (like desktop icons) can be small, you should propose 1 to 4 candidate region boxes that likely contain the target.
These boxes will be cropped and searched with high precision later.
Ensure the bounding boxes are large enough to contain the element and its direct surroundings (e.g., representing roughly a 200x200 to 400x400 physical pixel crop on a 1920x1080 screen).

All box coordinates (x_min, y_min, x_max, y_max) MUST be normalized between 0.0 and 1.0 relative to the full screen image.
- (0.0, 0.0) is the top-left corner.
- (1.0, 1.0) is the bottom-right corner.

You MUST output your response strictly as a JSON object with these keys:
{
  "candidates": [
    {
      "x_min": float,      // Minimum X (0.0 to 1.0)
      "y_min": float,      // Minimum Y (0.0 to 1.0)
      "x_max": float,      // Maximum X (0.0 to 1.0)
      "y_max": float,      // Maximum Y (0.0 to 1.0)
      "description": str,  // Rationale for this candidate region
      "confidence": float  // Estimated probability (0.0 to 1.0) that the element is here
    }
  ],
  "visual_clues": str      // High-level visual description of where the element is relative to landmarks
}
Do NOT include any markdown code blocks, backticks, or other text outside of the JSON object.
"""

USER_PROMPT_TEMPLATE = "Find all possible candidate regions for the UI element described as: '{instruction}' on this screen."

class Planner:
    """
    Planner module responsible for global screenshot analysis, identifying visual context,
    and proposing candidate search regions (bounding box crops).
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    def propose_candidate_regions(self, screenshot: Image.Image, instruction: str) -> Dict[str, Any]:
        """
        Sends full screenshot and instructions to the vision model.
        Returns proposed candidate bounding boxes (normalized).
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(instruction=instruction)
        
        logger.info(f"Planner proposing candidate search regions for: '{instruction}'...")
        
        try:
            response_text = self.client.call_vision_api(
                image=screenshot,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_response=True
            )
            
            parsed = self._parse_json_from_response(response_text)
            candidates = parsed.get("candidates", [])
            
            logger.info(f"Planner proposed {len(candidates)} candidate regions.")
            logger.debug(f"Visual Clues: {parsed.get('visual_clues', 'None')}")
            
            if not candidates:
                logger.warning("Planner returned empty candidate list! Activating visual search fallbacks.")
                return self._get_fallback_proposal()
                
            # Bounds check coordinates
            for c in candidates:
                c["x_min"] = max(0.0, min(1.0, float(c.get("x_min", 0.0))))
                c["y_min"] = max(0.0, min(1.0, float(c.get("y_min", 0.0))))
                c["x_max"] = max(0.0, min(1.0, float(c.get("x_max", 1.0))))
                c["y_max"] = max(0.0, min(1.0, float(c.get("y_max", 1.0))))
                
            MIN_BOX_SIZE = 0.05
            if (c["x_max"] - c["x_min"]) < MIN_BOX_SIZE:
                # Try expanding max first, then pull min back if at edge
                if c["x_max"] < 1.0 - MIN_BOX_SIZE:
                    c["x_max"] = c["x_min"] + MIN_BOX_SIZE
                else:
                    c["x_min"] = max(0.0, c["x_max"] - MIN_BOX_SIZE)

            if (c["y_max"] - c["y_min"]) < MIN_BOX_SIZE:
                if c["y_max"] < 1.0 - MIN_BOX_SIZE:
                    c["y_max"] = c["y_min"] + MIN_BOX_SIZE
                else:
                    c["y_min"] = max(0.0, c["y_max"] - MIN_BOX_SIZE)
                    
            return parsed
            
        except Exception as e:
            logger.error(f"Planner failed to propose candidates: {e}. Falling back to default search sectors.")
            return self._get_fallback_proposal()

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

    def _get_fallback_proposal(self) -> Dict[str, Any]:
        """
        Provides fallback search regions covering common areas (e.g. desktop icon zones).
        """
        logger.info("Generating robust search quadrant defaults.")
        return {
            "candidates": [
                # Sector 1: Left-aligned icons columns (most common)
                {
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 0.25,
                    "y_max": 0.95,
                    "description": "Fallback: Default Left Desktop Column Zone",
                    "confidence": 0.5
                },
                # Sector 2: Right-aligned icons column / status zone
                {
                    "x_min": 0.75,
                    "y_min": 0.0,
                    "x_max": 1.0,
                    "y_max": 0.95,
                    "description": "Fallback: Default Right Desktop Zone",
                    "confidence": 0.3
                },
                # Sector 3: Center focus region
                {
                    "x_min": 0.25,
                    "y_min": 0.25,
                    "x_max": 0.75,
                    "y_max": 0.75,
                    "description": "Fallback: Default Center Focus Zone",
                    "confidence": 0.2
                }
            ],
            "visual_clues": "Fallback quadrants used due to visual planner service disruption."
        }
