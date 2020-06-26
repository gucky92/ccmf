"""
Template for a possible redesign
"""

from typing import Union, Optional, Callable
from dataclasses import dataclass  # requires Python 3.7 or above

import networkx as nx
import torch
import pyro
from pyro.distributions import dist


GAIN_PREFIX = "gain_"
OFFSET_PREFIX = "offset_"
DATA_PREFIX = "data_"
SIGN_PREFIX = "sign_"


class _NonlinMixin:

    def sample_gain(self):
        """
        Sample nonlinearity gain
        """
        if self.nonlin is None or self.gain is None:
            return 1.0
        else:
            return pyro.sample(f"{GAIN_PREFIX}{self.name}", self.gain)

    def sample_offset(self):
        """
        Sample nonlinearity offset
        """
        if self.nonlin is None or self.offset is None:
            return 0.0
        else:
            return pyro.sample(f"{OFFSET_PREFIX}{self.name}", self.offset)

    def __call__(self, tensor):
        """
        Applies nonlinearity
        """

        if self.nonlin is None:
            return tensor
        else:
            offset = self.sample_offset()
            gain = self.sample_gain()
            return self.nonlin(
                gain * tensor - offset
            ) + self.nonlin(offset)


@dataclass
class Neuron(_NonlinMixin):
    """
    Neuron class that holds the following attributes
    """
    __slots__ = (
        'name', 'prior_dist', 'nonlin', 'gain', 'offset', 'vary'
    )
    name: str
    # prior distribution of the neuron
    prior_dist: dist.Normal = dist.Normal(0, 1)
    # intrinsic nonlinearity
    nonlin: Optional(Callable) = None
    # the gain prior if nonlinearity is given - this specifies the prior
    gain: Optional(dist.LogNormal) = dist.LogNormal(0, 1)
    # offset to fit if nonlinearity is given - this specifies the prior
    offset: Optional(dist.Normal) = dist.Normal(0, 1)
    # if a latent variable or not
    vary: bool = True

    def sample(self, mean, scale=1):
        # TODO dealing with scale
        # applies nonlinearity (if present)
        mean = self(mean)
        if self.vary:
            return pyro.sample(
                self.name, dist.Normal(mean, scale)
            )
        else:
            return mean

    def _mean(self, data=None):
        """
        Parameters
        ----------
        data : NeuronData instance
        """
        # TODO shape of tensors returned
        # current mean value
        if self.vary:
            # TODO what if param doesn't exist yet?
            # -> implement some init param method?
            return pyro.param(self.name)
        # if no data exist and data is None
        # (this could lead to error downstream)
        elif data is None:
            return self.prior_dist.mean
        else:
            return data.mean


@dataclass
class Synapse:
    """
    Synapse class that holds the following attributes
    """
    __slots__ = ('postsynaptic', 'presynaptic', 'prior_dist', 'sign', 'vary')
    # name of postsynaptic neuron
    postsynaptic: str
    # name of presynaptic neuron
    presynaptic: str
    # prior distribution
    prior_dist: dist.LogNormal = dist.LogNormal(0, 1)
    # sign of synapse (defaults to bernoulli)
    sign: Union[float, dist.Bernoulli] = dist.Bernoulli(0.5)
    # if a latent variable or not (i.e. fixed to mean of prior_dist)
    vary: bool = True

    @property
    def name(self):
        return f"{self.postsynaptic}_{self.presynaptic}"

    def sample_sign(self):
        """
        Sample the sign for the synapse
        """
        if isinstance(self.sign, dist.Bernoulli):
            sign = pyro.sample(
                f"{SIGN_PREFIX}{self.name}", self.sign
            )
            if sign:
                return 1.0
            else:
                return -1.0
        else:
            return sign

    def sample(self):
        """
        Sample the (signed) weight
        """
        s = self.sample_sign()
        if self.vary:
            w = pyro.sample(
                self.name, self.prior_dist
            )
            return s * w
        else:
            return s * self.prior_dist.mean


@dataclass
class NeuronData(_NonlinMixin):
    """
    Dataclass that holds the following attributes
    """

    def __post_init__(self):
        self.mask = ~torch.isnan(self.data)
        # is this the correct type for mathematical operations
        self.mask_float = self.mask.type(torch.float32)
        # create nanless data
        data = self.data.clone()
        data[~self.mask] = 0
        self.nanless = data
        n = torch.sum(self.mask_float, axis=1)[:, None]
        self.mean = torch.sum(self.nanless, axis=1)[:, None] / n
        # nanvar
        var = torch.sum(
            ((self.nanless - self.mean) ** 2) * self.mask_float,
            axis=1
        )[:, None] / n
        self.scale = var

    __slots__ = ('neuron_name', 'data', 'nonlin', 'gain', 'offset')
    # name of parameter
    neuron_name: str
    # data tensor
    data: torch.tensor
    # extrinsic nonlinearity
    nonlin: Optional(Callable) = None
    # the gain prior if nonlinearity is given - this specifies the prior
    gain: Optional(dist.Distribution) = None
    # offset to fit if nonlinearity is given - this specifies the prior
    offset: Optional(dist.Distribution) = dist.Normal(0, 1)

    @property
    def name(self):
        return f"{DATA_PREFIX}{self.neuron_name}"

    def sample(self, mean):
        # takes the empirical variance
        mean = self(mean)
        data, mean = torch.broadcast_tensors(self.nanless, mean)
        # normalize mean predictions
        # TODO specify l1 or l2 normalization
        mean = mean * (
            torch.sqrt(torch.sum(data ** 2 * self.mask_float, axis=0))[None]
            / torch.sqrt(torch.sum(mean ** 2 * self.mask_float, axis=1))[None]
        )
        # uses the empirical std in the normal
        # TODO is this how to apply the mask for observations?
        with pyro.poutine.mask(mask=self.mask):
            s = pyro.sample(
                self.name, dist.Normal(mean, self.scale), obs=data
            )
        return s


class CircuitModel:
    """
    Circuit model class that implements `conditioned_model` method.

    Parameters
    ----------
    circuit : networkx.DiGraph
    circuit_kwargs : dict
        Used to create Neurona and Synapse classes already in `circuit`.
    """

    def __init__(
        self,
        circuit: Optional(nx.DiGraph) = None,
        **circuit_kwargs  # TODO
    ):
        if circuit is None:
            self._circuit = nx.DiGraph()
        else:
            assert isinstance(circuit, nx.DiGraph)
            self._circuit = circuit

        self._neurons = {}
        # name : Neuron instance dictionary

        self._synapses = {}
        # (postsynaptic, presynaptic) : Synapse instance dictionary
        # synapse dictionary should have a two-tuple as a key

        self._data = {}  # name : Data instance dictionary
        # for the data dictionary the name should have the same names as
        # the ones that exist in neurons (see add_data)

    @property
    def circuit(self):
        return self._circuit

    @property
    def neurons(self):
        return self._neurons

    @property
    def synapses(self):
        return self._synapses

    @property
    def data(self):
        return self._data

    def add_neuron(
        name, prior_dist=None, nonlin=None, gain=None,
        offset=None, vary=None
    ):
        """
        Adds node to nx.DiGraph (if not exists) and creates Neuron instance
        """

    def add_synapse(
        postsynaptic, presynaptic, prior_dist=None,
        sign=None, vary=None
    ):
        """
        Adds edge to nx.DiGraph (if not exists) and creates Synapse instance
        """

    def add_data(
        neuron_name, data, nonlin=None, gain=None, offset=None
    ):
        """
        Add data to circuit model and create NeuronData instance.

        All data within one circuit have to have the same length
        (i.e. same set of stimuli used.)
        """

    def _init_pyro_param_store(self):
        """
        probably a private method that needs to be implemented to use with
        AutoGuide.
        """

    def inputs(self, neuron):
        """Return list of inputs to a neuron
        """

        return list(self.circuit[neuron])

    def conditioned_model(self):
        """Model for pyro
        """

        for name, neuron in self.neurons.items():
            inputs = self.inputs(name)

            if not inputs:
                # TODO
                continue

            for idx, input in enumerate(inputs):
                w_ij = self.synapses[(neuron, input)].sample()
                x_j = self.neurons[input]._mean(self.data.get(input, None))

                if idx == 0:
                    x = x_j[:, None]
                    w = w_ij[None]
                else:
                    x = torch.cat([x, x_j[:, None]], axis=1)
                    w = torch.cat([w, w_ij[None]], axis=0)

            # normalize w
            # TODO specify L1 or L2 normalization
            w = w / torch.sum(torch.abs(w))
            # normalize x
            # TODO specify L1 or F-norm
            x = x / torch.norm(x)
            # linear prediction
            y_pred = x @ w
            y_pred = neuron.sample(y_pred, 1)  # TODO dealing with scaling

            data = self.data.get(neuron, None)

            if data is not None:
                data.sample(y_pred)
