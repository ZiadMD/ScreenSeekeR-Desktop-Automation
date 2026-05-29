import json
import re
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from src.grounding.llm_client import LLMClient
from src.grounding.screenshot import capture_screen, physical_to_logical
from src.automation.desktop import click_at
from src.utils.logging import logger

SYSTEM_PROMPT = """
You are a desktop automation watchdog assistant. Your role is to examine the provided screenshot and detect any unexpected popups, modal dialogs, error boxes, system updates, or notifications that are blocking the normal desktop workspace or active application window.

If an unexpected popup or blocking dialog is detected:
1. Identify the dismissive element (like an 'X' close button, 'Cancel', 'Close', 'No', 'OK', or 'Don't Save').
2. Predict the center coordinate (x, y) of that button.
3. Normalize coordinates between 0.0 and 1.0 relative to the image dimensions.

You MUST output your response strictly as a JSON object:
{
  "popup_detected": bool,
  "dismiss_button_coords": {
    "x": float,       // Center X coordinate (0.0 to 1.0)
    "y": float        // Center Y coordinate (0.0 to 1.0)
  },
  "button_label": str, // E.g., 'Cancel', 'Close button', etc.
  "reasoning": str     // Brief visual explanation of the detected popup
}

If NO blocking popups, dialogs, or overlays are present and the desktop/active app is clear, return:
{
  "popup_detected": false
}

Do NOT include any markdown code blocks, backticks, or other text outside of the JSON object.
"""

USER_PROMPT = "Analyze the screenshot for any unexpected blocking popups or dialogs. If found, locate the close or cancel button."

class PopupHandler:
    """
    Zero-shot, vision-based detector and dismisser of unexpected popups, notifications,
    and dialogs. Uses vision API to locate and dismiss obstacles without template matching.
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    def check_and_dismiss_popups(self) -> bool:
        """
        Takes a screenshot, checks for blocking popups, and clicks their dismiss button if found.
        Returns:
            - True if a popup was detected and dismissed.
            - False if no popups were found.
        """
        logger.info("Taking watchdog screenshot to check for unexpected popups...")
        screenshot = capture_screen()
        phys_w, phys_h = screenshot.size
        
        try:
            response_text = self.client.call_vision_api(
                image=screenshot,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
                json_response=True
            )
            
            parsed = self._parse_json(response_text)
            
            if parsed.get("popup_detected", False):
                reason = parsed.get("reasoning", "Unknown blocking dialog")
                label = parsed.get("button_label", "Dismiss")
                coords = parsed.get("dismiss_button_coords", {})
                
                if "x" in coords and "y" in coords:
                    px = float(coords["x"]) * phys_w
                    py = float(coords["y"]) * phys_h
                    
                    # Convert physical coordinate to PyAutoGUI logical coordinate
                    lx, ly = physical_to_logical(px, py)
                    
                    logger.warning(f"Watchdog DETECTED a blocking popup: '{reason}'. Attempting to dismiss by clicking '{label}' at logical ({lx}, {ly})...")
                    click_at(lx, ly)
                    logger.info("Popup dismissal command executed.")
                    return True
                else:
                    logger.error("Popup detected, but coordinates were not provided in response.")
            else:
                logger.info("No unexpected popups or blocking dialogs detected. Desktop is clear.")
                
        except Exception as e:
            logger.error(f"Error checking or dismissing popups: {e}")
            
        return False

    def _parse_json(self, text: str) -> Dict[str, Any]:
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
