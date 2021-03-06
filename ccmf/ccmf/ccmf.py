import pandas as pd
from pyro.infer import NUTS, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam

from ccmf.inference import InferenceEngine
from ccmf.model.base import Model


class CCMF(InferenceEngine):
    """Circuit-constrained matrix factorization.

    """
    def __init__(self, model: Model, guide=AutoDelta, optimizer=Adam, loss=Trace_ELBO, kernel=NUTS, **options):
        """

        Parameters
        ----------
        model
        guide
        optimizer
        loss
        kernel
        options
        """
        self._model = model
        self._samples = None
        super().__init__(self._model.conditioned_model, guide, optimizer, loss, kernel, **options)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_mcmc']
        del state['_optimizer']
        return state

    def __setstate__(self, state):
        state['_optimizer'] = Adam
        self.__dict__ = state

    def fit(self, X: pd.DataFrame):
        """Do MAP Estimation.

        Parameters
        ----------
        X
            Data.
        Returns
        -------
        self
            Fitted estimator.
        """
        super().fit(self._model.preprocess(X))
        return self

    def run_mcmc(self, X: pd.DataFrame, initial_params='map', **options):
        """Run MCMC.

        Parameters
        ----------
        X
            Data.
        initial_params
            Initial state for MCMC algorithm. MAP estimates will be used if set to 'map'.

        options

        Returns
        -------
        self
        """
        if isinstance(initial_params, str) and initial_params.lower() == 'map':
            initial_params = self.map_estimates
        super().run_mcmc(self._model.preprocess(X), initial_params=initial_params, **options)
        self._samples = super().get_samples()
        return self

    def get_samples(self):
        """Get posterior samples generated by MCMC.

        Returns
        -------

        """
        return self._samples
