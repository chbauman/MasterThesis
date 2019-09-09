from abc import ABC, abstractmethod
 
class BaseDynamicsModel(ABC):
    """
    This class describes the interface of a ML-based
    dynamics model.
    """
    
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    def analyze(self):
        """
        Analyzes the trained model
        """
        print("Analyzing model")
