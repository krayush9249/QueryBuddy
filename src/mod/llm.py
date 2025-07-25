from langchain_together import TogetherLLM

def setup_llm(together_api_key: str, 
              model_name: str = "meta-llama/Llama-3-70b-chat-hf",
              temperature: float = 0.1) -> TogetherLLM:
    """Initialize the Together AI LLM"""
    return TogetherLLM(
        model=model_name,
        together_api_key=together_api_key,
        temperature=temperature,
        max_tokens=2000
    )