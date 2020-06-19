from abc import ABC, abstractmethod

from ccmf.circuit import Circuit
import pandas as pd


class Model(ABC):
    """Abstract Base Class for circuit models used in CCMF.

    """
    eps = 1e-12

    def __init__(self, circuit: Circuit):
        self._circuit = circuit
        self._mask = None
        self.prior = {}
        self.df_format = {}

    @abstractmethod
    def model(self):
        """A Pyro model.

        Returns
        -------

        """
        pass

    @abstractmethod
    def conditioned_model(self, X):
        """Conditioning `Model.model` with data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------

        """
        pass

    @abstractmethod
    def _init_prior(self):
        """Construct prior distributions based on `self._circuit`.

        Returns
        -------

        """
        pass

    @abstractmethod
    def preprocess(self, X: pd.DataFrame):
        """Preprocess data and adjust attributes according to the data.

        The method is called by :func:`~ccmf.ccmf.CCMF.fit` to preprocess the data to a format (e.g. from
        `pd.DataFrame` to dict of `torch.tensors`) that can be interpreted by the `model`. The method also adjusts the
        dimensions of some distributions (e.g. distributions whose dimension depends on the stimuli, since the number
        of stimuli is unknown before fitting the data). Finally, the method takes care of the nan values in X, which
        represents unobserved data. When an entry in X is nan, the corresponding entry in `self._mask` will be set to
        `False` so that Pyro treats it as an unobserved entry instead of causing unexpected behaviors.

        Parameters
        ----------
        X
            Response data of cells

        Returns
        -------

        """
        pass

    @abstractmethod
    def process_samples(self, samples):
        return samples
