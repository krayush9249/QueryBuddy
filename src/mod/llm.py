from langchain_together import TogetherLLM

def setup_llm(together_api_key: str, 
              model_name: str = "deepseek-ai/deepseek-coder-33b-instruct",
              temperature: float = 0.1) -> TogetherLLM:
    """Initialize the Together AI LLM"""
    return TogetherLLM(
        model=model_name,
        together_api_key=together_api_key,
        temperature=temperature,
        max_tokens=2000
    )