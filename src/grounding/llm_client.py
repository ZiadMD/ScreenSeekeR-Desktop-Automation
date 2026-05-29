import base64
import io
from typing import Optional, Literal
from PIL import Image
from src.config import settings
from src.utils.logging import logger

class LLMClient:
    """
    Unified, provider-agnostic client wrapper for Google Gemini, OpenAI, Groq, and Ollama APIs.
    Specifically designed for Vision models to support desktop grounding.
    """
    def __init__(
        self,
        provider: Optional[Literal["gemini", "openai", "groq", "ollama"]] = None,
        model_name: Optional[str] = None
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model_name = model_name or settings.GROUNDER_MODEL
        self._init_client()

    def _init_client(self):
        logger.info(f"Initializing LLMClient for provider: {self.provider} using model: {self.model_name}")
        
        if self.provider == "gemini":
            import google.generativeai as genai
            if not settings.GEMINI_API_KEY:
                logger.error("GEMINI_API_KEY is not set in .env! Gemini operations will fail.")
                raise ValueError("GEMINI_API_KEY is not set in .env")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.client = genai
            
        elif self.provider == "openai":
            from openai import OpenAI
            if not settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY is not set in .env! OpenAI operations will fail.")
                raise ValueError("OPENAI_API_KEY is not set in .env")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
        elif self.provider == "groq":
            from groq import Groq
            if not settings.GROQ_API_KEY:
                logger.error("GROQ_API_KEY is not set in .env! Groq operations will fail.")
                raise ValueError("GROQ_API_KEY is not set in .env")
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            
        elif self.provider == "ollama":
            import ollama
            # Setup custom host client
            self.client = ollama.Client(host=settings.OLLAMA_API_URL)
            
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def call_vision_api(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        json_response: bool = False
    ) -> str:
        """
        Calls the vision model with the provided image, system prompt, and user prompt.
        Optionally requests a JSON format response.
        """
        logger.debug(f"Calling vision API ({self.provider}/{self.model_name}) with image size: {image.width}x{image.height}")
        
        try:
            if self.provider == "gemini":
                return self._call_gemini(image, system_prompt, user_prompt, json_response)
            elif self.provider == "openai":
                return self._call_openai(image, system_prompt, user_prompt, json_response)
            elif self.provider == "groq":
                return self._call_groq(image, system_prompt, user_prompt, json_response)
            elif self.provider == "ollama":
                return self._call_ollama(image, system_prompt, user_prompt, json_response)
        except Exception as e:
            logger.error(f"Error calling vision API for provider {self.provider}: {e}")
            raise

    def _call_gemini(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        json_response: bool
    ) -> str:
        generation_config = {}
        if json_response:
            generation_config["response_mime_type"] = "application/json"
            
        # Combine system prompt and user prompt
        model = self.client.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt
        )
        
        response = model.generate_content(
            [image, user_prompt],
            generation_config=generation_config
        )
        
        if not response.text:
            raise ValueError("Empty response received from Gemini API.")
        return response.text

    def _call_openai(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        json_response: bool
    ) -> str:
        # Encode image to base64 jpeg
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                }
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload}
            ],
            response_format={"type": "json_object"} if json_response else None,
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from OpenAI API.")
        return content

    def _call_groq(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        json_response: bool
    ) -> str:
        # Encode image to base64 jpeg
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                }
            }
        ]
        
        # Groq doesn't consistently support response_format json on all vision models,
        # but supports the model instructions well.
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload}
            ],
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from Groq API.")
        return content

    def _call_ollama(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        json_response: bool
    ) -> str:
        # Encode image to bytes
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [img_bytes]}
            ],
            format="json" if json_response else None
        )
        
        content = response['message']['content']
        if not content:
            raise ValueError("Empty response received from Ollama API.")
        return content
