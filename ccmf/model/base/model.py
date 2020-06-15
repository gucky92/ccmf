from abc import ABC, abstractmethod


class Model(ABC):
    eps = 1e-12

    def __init__(self, circuit):
        self._circuit = circuit

    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def prior(self):
        pass

    @abstractmethod
    def preprocess(self, X):
        pass
