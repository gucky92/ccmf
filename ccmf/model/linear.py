from ccmf.circuit.sign import *
from .base import LinearRecurrent

import pyro.distributions as dist
import torch


class LinearRecurrent1(LinearRecurrent):
    def __init__(self, circuit, sigma_x=.01, sigma_u=1):
        super().__init__(circuit, sigma_x)
        self.sigma_u = sigma_u
        self.prior = self._init_prior()

    def _init_prior(self):
        circuit = self._circuit
        inputs, outputs = circuit.inputs, circuit.outputs

        sign_to_range = {EXCITATORY: (0, 1),
                         INHIBITORY: (-1, 0),
                         UNSPECIFIED: (-1, 1)}

        W = torch.zeros(len(outputs), len(inputs), 2)
        W[..., 1] += self.eps

        for i, u in enumerate(inputs):
            for j, v in enumerate(outputs):
                if circuit.has_edge(u, v):
                    W[j, i] = torch.tensor(sign_to_range[circuit.edges[u, v]['sign']])

        M = torch.zeros(len(outputs), len(outputs), 2)
        M[..., 1] += self.eps

        for i, u in enumerate(outputs):
            for j, v in enumerate(outputs):
                if circuit.has_edge(u, v):
                    M[j, i] = torch.tensor(sign_to_range[circuit.edges[u, v]['sign']])

        return {'W': dist.Uniform(*W.permute(2, 0, 1)),
                'M': dist.Uniform(*M.permute(2, 0, 1)),
                'U': dist.Normal(0, self.sigma_u)}

    def preprocess(self, X):
        df_X = [X.loc[i] for i in self._circuit.inputs + self._circuit.outputs]
        self._mask = [~torch.tensor(df_Xi.isna().values) for df_Xi in df_X]
        self.prior['U'] = self.prior['U'].expand(torch.Size([len(self._circuit.inputs), X.shape[1]]))
        return {f'X{i}': torch.tensor(df_Xi.fillna(0).values).float() for i, df_Xi in enumerate(df_X)}
