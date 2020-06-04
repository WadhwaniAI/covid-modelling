from abc import ABC, abstractmethod
from models import Model

class SEIR(Model):

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
