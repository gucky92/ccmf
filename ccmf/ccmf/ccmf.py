from pyro.infer import NUTS, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam

from ccmf.inference import InferenceEngine


class CCMF(InferenceEngine):
    def __init__(self, circuit, generative, guide=AutoDelta, optimizer=Adam, loss=Trace_ELBO, kernel=NUTS, **options):
        super().__init__(generative(circuit).model, guide, optimizer, loss, kernel, **options)
