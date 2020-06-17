from pyro.infer import NUTS, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam

from .map import MAPEstimator
from .mcmc import MCMCSampler


class InferenceEngine(MAPEstimator, MCMCSampler):
    """Inference class supporting MAP estimation and MCMC sampling.

    """
    def __init__(self, model, guide=AutoDelta, optimizer=Adam, loss=Trace_ELBO, kernel=NUTS, **options):
        MAPEstimator.__init__(self, model, guide, optimizer, loss, **options)
        MCMCSampler.__init__(self, model, kernel, **options)
