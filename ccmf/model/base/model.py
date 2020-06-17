from abc import ABC, abstractmethod


class Model(ABC):
    eps = 1e-12

    def __init__(self, circuit):
        self._circuit = circuit
        self._mask = None
        self.prior = {}
        self.df_format = {}

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def conditioned_model(self, X):
        pass

    @abstractmethod
    def _init_prior(self):
        pass

    @abstractmethod
    def preprocess(self, X):
        pass
