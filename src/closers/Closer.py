from abc import ABC, abstractmethod

class Closer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass