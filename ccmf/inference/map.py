import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from sklearn.base import BaseEstimator
from tqdm import tqdm


class MAPEstimator(BaseEstimator):
    def __init__(self, model, guide=AutoDelta, optimizer=Adam, loss=Trace_ELBO, **options):
        self.__model = model
        self._guide = guide(model) if isinstance(guide, type) else guide
        self._optimizer = optimizer
        self._loss = loss() if isinstance(loss, type) else loss
        self.__defaults = {'lr': 1e-3, 'max_iter': 1000}
        self.__defaults.update(options)
        self._loss_curve = None
        self._svi = None

    def fit(self, *args, **options):
        params = self.__defaults.copy()
        params.update(options)
        pyro.clear_param_store()
        optimizer = self._optimizer({'lr': params['lr']})

        self._guide(*args)
        self._svi = SVI(self.__model, self._guide, optimizer, self._loss)
        self._loss_curve = [self._svi.step(*args) for _ in tqdm(range(params['max_iter']))]

    @property
    def loss_curve(self):
        return self._loss_curve

    @property
    def map_estimates(self):
        if self._svi:
            return {latent: median.detach() for latent, median in self._guide.median().items()}
