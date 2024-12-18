from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List

class LLMConfig(BaseModel):
    llm_name: str
    top_p: float = 0.1
    temperature: float = 0.3
    stream: bool = False
    incremental_output: bool = False

class LLM(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def chat(self, input_prompt: str) -> str:
        pass

    @abstractmethod
    def multi_chat(
        self,
        messages: List[dict],
        user_content,
        pure_user_content,
        use_for_history: bool = True,
    ):
        pass
