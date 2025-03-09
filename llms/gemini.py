import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os

class GeminiLLM:
    """Wrapper for Gemini LLM models."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        """
        Initialize the Gemini LLM wrapper.
        
        Args:
            model_name: Name of the Gemini model to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash")
            api_key: Google API key. If None, it will try to get it from the environment variable GOOGLE_API_KEY.
        """
        self.model_name = model_name
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set as GOOGLE_API_KEY environment variable")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Default generation parameters for different models
        # These will be used when generating content, not when initializing the model
        self.generation_params = {
            "gemini-1.5-pro": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
            "gemini-1.5-flash": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
            "gemini-1.5-flash": {"temperature": 0.7, "top_p": 0.95, "top_k": 40}
        }
        
        # Validate model name
        if model_name not in self.generation_params:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.generation_params.keys())}")
        
        # Initialize the model without passing generation parameters
        self.model = genai.GenerativeModel(model_name)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the Gemini model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to guide the model's behavior
            **kwargs: Additional parameters to pass to the model
        
        Returns:
            The model's response as a string
        """
        # Prepare generation config with model-specific defaults
        generation_config = {}
        generation_config.update(self.generation_params[self.model_name])
        
        # Override with any user-specified parameters
        generation_config.update(kwargs)
        
        try:
            # Handle system prompt in a different way
            if system_prompt:
                # For Gemini models, we can incorporate the system prompt directly into the user prompt
                # since some versions don't support system_instruction
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
                response = self.model.generate_content(full_prompt, generation_config=generation_config)
            else:
                response = self.model.generate_content(prompt, generation_config=generation_config)
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}" 