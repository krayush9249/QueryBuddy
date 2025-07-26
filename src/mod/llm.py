from typing import List, Optional
import together
from langchain_together import Together
from langchain_core.language_models import LLM


class LLMSetup(LLM):
    model_name: str
    together_api_key: str
    temperature: float = 0.1
    max_tokens: int = 2000

    def __init__(self,
                 together_api_key: str,
                 model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """
        Custom LangChain-compatible LLM wrapper around Together.
        """
        super().__init__(
            together_api_key=together_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Create a Together client instance
        self._llm = Together(
            model=model_name,
            together_api_key=together_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )

    @property
    def _llm_type(self) -> str:
        return "together_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Required method to make the class work with LangChain.
        """
        return self._llm.invoke(prompt)

# class LLMSetup(LLM):
#     def __init__(self,
#                  together_api_key: str,
#                  model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#                  temperature: float = 0.1,
#                  max_tokens: int = 2000):
#         """
#         Custom LangChain-compatible LLM wrapper around the Together AI client.
#         """
#         # Setup the Together API key
#         self.model = model_name
#         self.temperature = temperature
#         self.max_tokens = max_tokens

#         together.api_key = together_api_key
#         self.client = together.Together()

#     @property
#     def _llm_type(self) -> str:
#         return "together_custom"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         """
#         Sends prompt to Together AI and returns the generated response.
#         """
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             stop=stop
#         )

#         return response.choices[0].message.content.strip()

# class LLMSetup(LLM):
#     def __init__(self,
#                  together_api_key: str,
#                  model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", # "deepseek-ai/deepseek-coder-33b-instruct",
#                  temperature: float = 0.1,
#                  max_tokens: int = 2000):
#         """
#         Custom LangChain-compatible LLM wrapper around Together.
#         """

#         self._llm = Together(
#             model=model_name,
#             together_api_key=together_api_key,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )

#     @property
#     def _llm_type(self) -> str:
#         return "together_custom"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         """
#         Required method to make the class work with LangChain.
#         """
#         return self._llm.invoke(prompt)