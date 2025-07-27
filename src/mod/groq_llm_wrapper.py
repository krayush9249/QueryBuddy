from langchain_groq import ChatGroq
from langchain_core.language_models import LLM
from typing import List, Optional, Dict, Any
import requests


class GroqLLM(LLM):
    """
    Custom LangChain-compatible LLM wrapper for Groq API.
    """
    
    def __init__(self,
                 groq_api_key: str,
                 model_name: str,
                 temperature: float = 0.1,
                 max_tokens: int = 5000,
                 top_p: float = 1.0):
        """
        Initialize Groq LLM wrapper.
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Validate API key
        if not groq_api_key:
            raise ValueError("Groq API key is required")

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "groq_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Make API call to Groq and return response.
        """
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False
        }
        
        # Add stop sequences if provided
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from Groq API: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling Groq API: {str(e)}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for this LLM."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }


# Simple setup function for dual LLMs with Groq
def setup_groq_llms(groq_api_key: str, 
                    coding_model: str = "llama3-70b-8192",
                    chat_model: str = "mixtral-8x7b-32768") -> (GroqLLM, GroqLLM):
    """
    Create both coding and chat LLMs using Groq models.
    Returns: (coding_llm, chat_llm)
    """
    # Coding LLM
    coding_llm = GroqLLM(
        groq_api_key=groq_api_key,
        model_name=coding_model,  
        temperature=0.1,  # Low temperature for deterministic code
        max_tokens=5000
    )
    
    # Chat LLM 
    chat_llm = GroqLLM(
        groq_api_key=groq_api_key,
        model_name=chat_model,  
        temperature=0.1, 
        max_tokens=5000
    )
    
    return coding_llm, chat_llm


# # Usage example
# if __name__ == "__main__":
#     # Replace with your actual Groq API key
#     api_key = "your_groq_api_key_here"
    
#     try:
#         # Setup both LLMs
#         coding_llm, chat_llm = setup_groq_llms(api_key)
        
#         # Test coding LLM
#         print("Testing Coding LLM...")
#         code_response = coding_llm.invoke("Write a Python function to calculate factorial")
#         print("Coding Response:", code_response)
#         print("\n" + "="*50 + "\n")
        
#         # Test chat LLM  
#         print("Testing Chat LLM...")
#         chat_response = chat_llm.invoke("Explain what recursion is in simple terms")
#         print("Chat Response:", chat_response)
#         print("\n" + "="*50 + "\n")
        
#         # Show available models
#         print("Available Groq Models:")
#         for model_id, model_name in GROQ_MODELS.items():
#             print(f"  {model_id}: {model_name}")
            
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Make sure you have a valid Groq API key!")