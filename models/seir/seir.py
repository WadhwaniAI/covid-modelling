from abc import ABC, abstractmethod

class SEIR(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_derivative(self):
        pass

    @abstractmethod
    def solve_ode(self):
        pass

    @abstractmethod
    def predict(self):
        pass
