from langchain_groq import ChatGroq

def setup_groq_llm(groq_api_key: str, 
                    model_name: str) -> ChatGroq:
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,  
        temperature=0.1, # Low temperature for deterministic code
        max_tokens=2000
    )
    return llm


from langchain_together import ChatTogether

def setup_together_chat_llm(together_api_key: str, 
                       model_name: str) -> ChatTogether:
    llm = ChatTogether(
        together_api_key=together_api_key,
        model=model_name,
        temperature=0.1,  # Low temperature for deterministic code
        max_tokens=2000
    )
    return llm


from langchain_together import Together

def setup_together_code_llm(together_api_key: str, 
                       model_name: str) -> Together:
    llm = Together(
        model=model_name,
        together_api_key=together_api_key,
        temperature=0.1,  # Low temperature for deterministic code
        max_tokens=2000
    )
    return llm