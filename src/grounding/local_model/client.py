"""
LLMClient-compatible wrapper for local vision models.

Bridges local model adapters (like GUI-Actor) into the same interface
used by the cloud-based providers (Gemini, OpenAI, Groq, Ollama),
so the Grounder and Planner can use them transparently.
"""

import json
import re
from typing import Optional
from PIL import Image
from src.utils.logging import logger


class LocalModelClient:
    """
    LLMClient-compatible wrapper for local vision models.
    
    Adapts local model output to match the call_vision_api() interface
    expected by the Grounder and Planner modules.
    """

    def __init__(
        self,
        model_type: str = "gui-actor",
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        attn_impl: str = "sdpa",
        max_pixels: int = 3200 * 1800,
    ):
        self.model_type = model_type

        if model_type == "gui-actor":
            from src.grounding.local_model.gui_actor_adapter import GUIActorAdapter
            self.adapter = GUIActorAdapter(
                model_path=model_path,
                device=device,
                torch_dtype=torch_dtype,
                attn_impl=attn_impl,
                max_pixels=max_pixels,
            )
        else:
            raise ValueError(
                f"Unknown local model type: '{model_type}'. "
                f"Supported types: 'gui-actor'"
            )

        logger.info(f"LocalModelClient initialized with model_type='{model_type}'")

    def call_vision_api(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        json_response: bool = False
    ) -> str:
        """
        Run local model inference and return results as a JSON string.
        
        For GUI-Actor, this extracts the instruction from the user_prompt,
        runs the adapter's native grounding method, and returns the result
        as a JSON string matching the Grounder's expected format.
        """
        # Extract the actual instruction from the user_prompt template
        instruction = self._extract_instruction(user_prompt)
        logger.debug(f"LocalModelClient inference: instruction='{instruction}'")

        # Run the adapter's grounding method
        result = self.adapter.ground_element(image, instruction)

        # Return as JSON string (matching what cloud APIs return)
        return json.dumps(result)

    def _extract_instruction(self, user_prompt: str) -> str:
        """
        Extract the target element description from the formatted user prompt.
        
        The Grounder/Planner format their prompts like:
          "Find the precise location of the UI element described as: 'Notepad' within this crop."
          "Find all possible candidate regions for the UI element described as: 'Notepad' on this screen."
        
        We extract the quoted instruction.
        """
        # Try to extract from quoted instruction pattern
        match = re.search(r"described as:\s*'([^']+)'", user_prompt)
        if match:
            return match.group(1)

        # Fallback: use the whole prompt
        return user_prompt
