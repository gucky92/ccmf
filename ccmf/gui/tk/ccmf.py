from pyro.infer import NUTS, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam

from ccmf.ccmf import CCMF
from .gui_mixin import CCMFGUIMixin


class GUICCMF(CCMF, CCMFGUIMixin):
    def __init__(self, generative, guide=AutoDelta, optimizer=Adam, loss=Trace_ELBO, kernel=NUTS, **options):
        CCMFGUIMixin.__init__(self)
        CCMF.__init__(self, self._circuit, generative, guide, optimizer, loss, kernel, **options)
