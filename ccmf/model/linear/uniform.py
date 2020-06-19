import pyro.distributions as dist
import torch

from ccmf.circuit.sign import *
from ccmf.circuit.circuit import Circuit
from .base import LinearRecurrent


class UniformModel(LinearRecurrent):
    """A linear recurrent model with uniform distributions as priors.

    """
    def __init__(self, circuit: Circuit, sigma_x=.01, sigma_u=1):
        super().__init__(circuit, sigma_x)
        self.sigma_u = sigma_u
        self.prior = self._init_prior()

    def _init_prior(self):
        """Construct uniform distributions based on the signs of connections.

        The distribution of an entry in the weight matrix is determined by the corresponding connection sign according
        to:
        * U(0, 1) if `EXCITATORY`
        * U(-1, 0) if `INHIBITORY`
        * U(-1, 1) if `UNSPECIFIED`

        Returns
        -------

        """
        circuit = self._circuit
        inputs, outputs = circuit.inputs, circuit.outputs

        sign_to_range = {EXCITATORY: (0, 1),
                         INHIBITORY: (-1, 0),
                         UNSPECIFIED: (-1, 1)}

        W = torch.zeros(len(outputs), len(inputs), 2)
        self.W_mask = torch.zeros(len(outputs), len(inputs))

        for i, u in enumerate(inputs):
            for j, v in enumerate(outputs):
                if circuit.has_edge(u, v):
                    W[j, i] = torch.tensor(sign_to_range[circuit.edges[u, v]['sign']])
                    self.W_mask[j, i] = 1
                else:
                    W[j, i] = torch.tensor((-1000, 1000))

        M = torch.zeros(len(outputs), len(outputs), 2)
        self.M_mask = torch.zeros(len(outputs), len(outputs))

        for i, u in enumerate(outputs):
            for j, v in enumerate(outputs):
                if circuit.has_edge(u, v):
                    M[j, i] = torch.tensor(sign_to_range[circuit.edges[u, v]['sign']])
                    self.M_mask[j, i] = 1
                else:
                    M[j, i] = torch.tensor((-1000, 1000))

        self.df_format = {
            'W': {'index': outputs, 'columns': inputs},
            'M': {'index': outputs, 'columns': outputs},
            'U': {'index': inputs},
            'X': {'index': inputs + outputs}
        }

        return {'W': dist.Uniform(*W.permute(2, 0, 1)),
                'M': dist.Uniform(*M.permute(2, 0, 1)),
                'U': dist.Normal(0, self.sigma_u)}
