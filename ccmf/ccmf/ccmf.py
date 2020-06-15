from pyro.infer import NUTS, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam

from ccmf.inference import InferenceEngine


class CCMF(InferenceEngine):
    def __init__(self, model, guide=AutoDelta, optimizer=Adam, loss=Trace_ELBO, kernel=NUTS, **options):
        self._model = model
        super().__init__(self._model.conditioned_model, guide, optimizer, loss, kernel, **options)

    def fit(self, X):
        return super().fit(self._model.preprocess(X))
