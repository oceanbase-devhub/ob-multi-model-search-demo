from typing import Optional
from abc import ABC, abstractmethod
from ..llm import LLM
from ..tools import Tool

class Agent(ABC):
    DEFAULT_RETRY_COUNT = 3

    def __init__(self, llm: LLM, tool: Optional[Tool] = None):
        self.llm = llm
        self.tool = tool
    
    @abstractmethod
    def chat(self, **kwargs):
        pass