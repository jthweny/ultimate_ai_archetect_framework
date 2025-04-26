import os
import logging
from typing import Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class with common functionality."""
    
    def __init__(self):
        """Initialize the base agent."""
        pass
        
    def _get_llm_client(self, model_name: str) -> Optional[Any]:
        """
        Get an LLM client based on the model name prefix.
        
        Args:
            model_name: The name of the model to use, with prefix indicating the provider.
                        Supported prefixes: 'openrouter.', 'google_gemini.'
        
        Returns:
            An LLM client instance or None if initialization fails.
        """
        try:
            # Handle OpenRouter models (using ChatOpenAI)
            if model_name.startswith("openrouter."):
                try:
                    from langchain_openai import ChatOpenAI
                except ImportError:
                    logger.error("Failed to import ChatOpenAI. Please install langchain-openai package.")
                    return None
                
                try:
                    # Get the actual model name (remove the prefix)
                    actual_model = model_name.replace("openrouter.", "")
                    
                    # Get API key and base URL from environment variables
                    api_key = os.environ.get("OPENAI_API_KEY")
                    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
                    
                    if not api_key:
                        logger.error("OPENAI_API_KEY environment variable not set")
                        return None
                    
                    # Create and return the ChatOpenAI client with correct parameters
                    # Note: No headers parameter as per requirements
                    return ChatOpenAI(
                        model=actual_model,
                        openai_api_key=api_key,
                        openai_api_base=api_base,
                        temperature=0.7,  # Default temperature, can be adjusted as needed
                        max_tokens=1024   # Default max_tokens, can be adjusted as needed
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize OpenRouter client: {str(e)}")
                    return None
            
            # Handle Google Gemini models
            elif model_name.startswith("google_gemini."):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                except ImportError:
                    logger.error("Failed to import ChatGoogleGenerativeAI. Please install langchain-google-genai package.")
                    return None
                
                try:
                    # Get the actual model name (remove the prefix)
                    actual_model = model_name.replace("google_gemini.", "")
                    
                    # Get API key from environment variables
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    
                    if not api_key:
                        logger.error("GOOGLE_API_KEY environment variable not set")
                        return None
                    
                    # Create and return the ChatGoogleGenerativeAI client
                    return ChatGoogleGenerativeAI(
                        model=actual_model,
                        google_api_key=api_key,
                        temperature=0.7,  # Default temperature, can be adjusted as needed
                        max_tokens=1024   # Default max_tokens, can be adjusted as needed
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize Google Gemini client: {str(e)}")
                    return None
            
            # Unsupported model prefix
            else:
                logger.error(f"Unsupported model prefix in '{model_name}'. Supported prefixes: 'openrouter.', 'google_gemini.'")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in _get_llm_client: {str(e)}")
            return None
