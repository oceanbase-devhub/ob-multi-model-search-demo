from abc import ABC, abstractmethod

class Tool(ABC):
    @abstractmethod
    def call(self, **kwargs):
        pass