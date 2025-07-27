from langchain_groq import ChatGroq
from typing import Tuple


def setup_groq_llms(groq_api_key: str, 
                    coding_model: str,
                    chat_model: str) -> Tuple[ChatGroq, ChatGroq]:
    """
    Create both coding and chat LLMs using Groq models.
    Returns: (coding_llm, chat_llm)
    """
    # Coding LLM
    coding_llm = ChatGroq(
        api_key=groq_api_key,
        model_name=coding_model,  
        temperature=0.1, # Low temperature for deterministic code
        max_tokens=5000
    )
    
    # Chat LLM 
    chat_llm = ChatGroq(
        api_key=groq_api_key,
        model_name=chat_model,  
        temperature=0.3,  # High temperature for creative chat responses
        max_tokens=5000
    )
    
    return coding_llm, chat_llm