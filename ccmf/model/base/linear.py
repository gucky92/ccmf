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
        EX = torch.stack([U, EV])
        pyro.sample('X', dist.Normal(EX, self.sigma_x))

    @staticmethod
    def EV(W, M, U):
        return torch.inverse(torch.eye(*M.shape) - M) @ W @ U
