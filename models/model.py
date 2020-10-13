from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        pass
