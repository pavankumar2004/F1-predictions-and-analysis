import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass