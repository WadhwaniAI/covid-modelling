from abc import ABC, abstractmethod

class BaseLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_data(self):
        pass