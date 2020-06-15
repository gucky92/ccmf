from abc import ABC

import pyro
import pyro.distributions as dist
import torch

from .model import Model


class LinearRecurrent(Model, ABC):
    def __init__(self, circuit, sigma_x):
        super().__init__(circuit)
        self.sigma_x = sigma_x

    def model(self):
        W = pyro.sample('W', self.prior['W'])
        M = pyro.sample('M', self.prior['M'])
        U = pyro.sample('U', self.prior['U'])
        EV = self.EV(W, M, U)
        EX = torch.cat([U, EV])

        for i, row in enumerate(EX):
            with pyro.poutine.mask(mask=self._mask[i]):
                pyro.sample(f'X{i}', dist.Normal(EX[i], self.sigma_x)).expand(self._mask[i].shape)

    def conditioned_model(self, X):
        return pyro.condition(self.model, data=X)()

    @staticmethod
    def EV(W, M, U):
        return torch.inverse(torch.eye(*M.shape) - M) @ W @ U
