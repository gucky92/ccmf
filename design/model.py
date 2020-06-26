"""
Template for a possible redesign
"""

from typing import Union, Optional, Callable
from dataclasses import dataclass  # requires Python 3.7 or above
from numbers import Number

import networkx as nx
import numpy as np
import torch
import pyro
import pyro.distributions as dist


GAIN_PREFIX = "gain_"
OFFSET_PREFIX = "offset_"
DATA_PREFIX = "data_"
SIGN_PREFIX = "sign_"
FLOAT_TYPE = torch.float32


class _NonlinMixin:

    def sample_gain(self):
        """
        Sample nonlinearity gain
        """
        return pyro.sample(f"{GAIN_PREFIX}{self.name}", self.gain)

    def sample_offset(self):
        """
        Sample nonlinearity offset
        """
        return pyro.sample(f"{OFFSET_PREFIX}{self.name}", self.offset)

    def __call__(self, tensor):
        """
        Applies nonlinearity
        """

        if self.nonlin is None:
            return tensor
        elif self.gain is None and self.offset is None:
            return self.nonlin(tensor)
        elif self.gain is None:
            offset = self.sample_offset()
            return self.nonlin(
                tensor - offset
            ) + self.nonlin(offset)
        elif self.offset is None:
            gain = self.sample_gain()
            return self.nonlin(gain * tensor)
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
    name: str
    # intrinsic nonlinearity
    nonlin: Optional[Callable] = None
    # the gain prior if nonlinearity is given - this specifies the prior
    gain: Optional[dist.LogNormal] = dist.LogNormal(0, 1)
    # offset to fit if nonlinearity is given - this specifies the prior
    offset: Optional[dist.Normal] = dist.Normal(0, 1)
    # if a latent variable or not
    vary: bool = True
    # scale
    scale: float = 1.0
    # TODO heteroscedasticity? - effect on prior_dist in _mean

    def sample(self, mean):
        # applies nonlinearity if present
        mean = self(mean)
        # if vary sample from normal otherwise just return mean
        if self.vary:
            return pyro.sample(self.name, dist.Normal(mean, self.scale))

        else:
            return mean

    def _data_sample(self, sample_size, neuron_data):
        """
        Parameters
        ----------
        sample_size : int
        neuron_data : NeuronData instance
        """
        if neuron_data is None:
            # no data use uninformative prior
            return dist.Normal(0, self.scale).sample((sample_size,))
        else:
            # draw from data mean and scale
            # if data scale is zero this is deterministic
            return dist.Normal(
                neuron_data.mean, neuron_data.scale
            ).sample()


@dataclass
class Synapse:
    """
    Synapse class that holds the following attributes
    """
    # TODO plugging in synaptic count and getting the correct LogNormal
    # name of presynaptic neuron
    presynaptic: str
    # name of postsynaptic neuron
    postsynaptic: str
    # prior distribution
    prior_dist: dist.LogNormal = dist.LogNormal(0, 1)
    # sign of synapse (defaults to bernoulli)
    sign: Union[float, dist.Bernoulli] = dist.Bernoulli(0.5)
    # if a latent variable or not (i.e. fixed to mean of prior_dist)
    vary: bool = True

    @property
    def name(self):
        return f"{self.presynaptic}->{self.postsynaptic}"

    def sample_sign(self):
        """
        Sample the sign for the synapse
        """
        if isinstance(self.sign, dist.Bernoulli):
            # TODO won't work with pyro
            sign = pyro.sample(f"{SIGN_PREFIX}{self.name}", self.sign)

            if sign:
                return 1.0
            else:
                return -1.0

        else:
            return self.sign

    def sample(self):
        """
        Sample the (signed) weight
        """
        s = self.sample_sign()

        if self.vary:
            w = pyro.sample(self.name, self.prior_dist)
            return s * w

        else:
            return s * self.prior_dist.mean


@dataclass
class NeuronData(_NonlinMixin):
    """
    Dataclass that holds the following attributes
    """

    # TODO assign fixed gain and offset dependent on data?

    def __post_init__(self):
        if isinstance(self.data, np.ndarray):
            self.data = torch.tensor(self.data, dtype=FLOAT_TYPE)

        self.mask = ~torch.isnan(self.data)
        # is this the correct type for mathematical operations
        self.mask_float = self.mask.type(FLOAT_TYPE)
        # create nanless data
        data = self.data.clone()
        data[~self.mask] = 0
        self.nanless = data
        n = torch.sum(self.mask_float, axis=1)
        # mean
        self.mean = torch.sum(self.nanless, axis=1) / n
        # unbiased variance
        # nanvar
        if self.scale is None:
            # default variance of 1 if not enough samples - TODO change?
            var = torch.ones(n.shape, dtype=FLOAT_TYPE)
            var[n > 1] = torch.sum(
                ((self.nanless - self.mean[:, None]) ** 2) * self.mask_float,
                axis=1
            )[n > 1] / (n - 1.0)[n > 1]
            self.scale = torch.sqrt(var)
        else:
            if isinstance(self.scale, Number):
                self.scale = torch.ones(n.shape, dtype=FLOAT_TYPE) * self.scale
            elif isinstance(self.scale, np.ndarray):
                self.scale = torch.tensor(self.scale, dtype=FLOAT_TYPE)
            assert self.scale.shape == n.shape

    # name of parameter
    neuron_name: str
    # data tensor
    data: Union[torch.tensor, np.ndarray]
    # extrinsic nonlinearity
    nonlin: Optional[Callable] = None
    # the gain prior if nonlinearity is given - this specifies the prior
    gain: Optional[dist.Distribution] = None
    # offset to fit if nonlinearity is given - this specifies the prior
    offset: Optional[dist.Distribution] = dist.Normal(0, 1)
    # population standard deviation (same length as data)
    scale: Optional[Union[torch.tensor, np.ndarray, float]] = None

    @property
    def name(self):
        return f"{DATA_PREFIX}{self.neuron_name}"

    def sample(self, mean):
        # takes the empirical variance
        mean = self(mean)
        # broadcast to normalize each observation
        data, mean = torch.broadcast_tensors(self.nanless, mean[:, None])
        # normalize mean predictions
        # TODO specify l1 or l2 normalization
        mean = mean * (
            torch.sqrt(torch.sum(data ** 2 * self.mask_float, axis=0))[None]
            / torch.sqrt(torch.sum(mean ** 2 * self.mask_float, axis=0))[None]
        )
        # uses the empirical std in the normal
        # TODO is this how to apply the mask for observations?
        with pyro.poutine.mask(mask=self.mask):
            s = pyro.sample(
                self.name, dist.Normal(mean, self.scale[:, None]),
                obs=self.data
            )

        return s


# TODO
# - get mean effective weight matrix after fitting
# - get mean latent_X after fitting
# - plotting functionality
# - generateing data
# - sampling from circuit after fitting (return dictionary of latents)
# - slicing of circuit data for splitting dataset and cross-validation
class CircuitModel:
    """
    Circuit model class that implements `conditioned_model` method.

    Parameters
    ----------
    circuit : networkx.DiGraph
    """

    def __init__(self, circuit: Optional[nx.DiGraph] = None):
        if circuit is None:
            self._circuit = nx.DiGraph()
        else:
            assert isinstance(circuit, nx.DiGraph)
            self._circuit = circuit

        # Neuron instances are stored in the `neuron` key of the circuit node
        # Synapse instances are stored in the `synapse` key of edge attributes
        # Data instances are stored in the `data` key of the circuit node

        # sample size ones data has been added
        self._sample_size = None
        # initialized latent X dictionary
        self._latent_X_dict = None

    @property
    def circuit(self):
        return self._circuit

    @property
    def neurons(self):
        """
        Dictionary with `Neuron` instances for each neuron.

        The keys are the names of the neurons.
        """
        return {
            key: value.get('neuron', Neuron(key))
            for key, value in self.circuit.nodes.items()
        }

    @property
    def synapses(self):
        """
        Dictionary with `Synapse` instances for each existing synapse.

        The keys are a two-tuple of form (presynaptic, postsynaptic)
        """
        return {
            key: value.get('synapse', Synapse(key[0], key[1]))
            for key, value in self.circuit.edges.items()
        }

    @property
    def data(self):
        """
        Dictionary with `NeuronData` instances for each neuron.

        The keys are the names of the neurons.

        Neurons with no data are still present in dictionary but have
        None as their value.
        """
        return {
            key: value.get('neuron_data', None)
            for key, value in self.circuit.nodes.items()
        }

    @property
    def sample_size(self):
        if self._sample_size is None:
            # TODO messaging if size error
            for node_name, neuron_data in self.data.items():
                if neuron_data is not None:
                    self._sample_size = neuron_data.data.shape[0]
                    break
        return self._sample_size

    def add_neuron(self, name, **neuron_kwargs):
        """
        Adds node to nx.DiGraph (if not exists) and creates Neuron instance
        """
        neuron = Neuron(name, **neuron_kwargs)
        self.circuit.add_node(name, neuron=neuron)
        return self

    def add_synapse(self, presynaptic, postsynaptic, **synapse_kwargs):
        """
        Adds edge to nx.DiGraph (if not exists) and creates Synapse instance
        """
        assert postsynaptic in self.circuit.nodes, (
            f"Postsynaptic neuron `{postsynaptic}` not in circuit. "
            "Specify neuron first with `add_neuron` method."
        )
        assert presynaptic in self.circuit.nodes, (
            f"Presynaptic neuron `{presynaptic}` not in circuit. "
            "Specify neuron first with `add_neuron` method."
        )
        synapse = Synapse(presynaptic, postsynaptic, **synapse_kwargs)
        self.circuit.add_edge(presynaptic, postsynaptic, synapse=synapse)
        return self

    def add_data(self, neuron_name, data, **data_kwargs):
        """
        Add data to circuit model and create NeuronData instance.

        Also sets sample size, if it is the first data entered.

        All data within one circuit have to have the same length
        (i.e. same set of stimuli used.)
        """
        assert neuron_name in self.circuit.nodes, (
            f"Neuron `{neuron_name}` not in circuit. "
            "Specify neuron first with `add_neuron` method."
        )
        if self.sample_size is None:
            self._sample_size = data.shape[0]
        else:
            assert data.shape[0] == self.sample_size, (
                f"Data size of {data.shape[0]} does not match sample "
                f"size of {self.samples_size} for circuit."
            )
        neuron_data = NeuronData(neuron_name, data, **data_kwargs)
        self.circuit.nodes[neuron_name].update({'neuron_data': neuron_data})
        return self

    def inputs(self, neuron):
        """
        Return list of inputs to a neuron
        """
        return [pre for pre, post in self.circuit.in_edges(neuron)]

    @property
    def _latent_X(self):
        if self._latent_X_dict is None:
            data = self.data
            neurons = self.neurons

            dictionary = {}
            for name, neuron in neurons.items():
                dictionary[name] = neuron._data_sample(
                    self.sample_size,
                    neuron_data=data[name]
                )

            self._latent_X_dict = dictionary

        return self._latent_X_dict

    def conditioned_model(self):
        """Model for pyro
        """

        neurons = self.neurons
        synapses = self.synapses
        data = self.data

        for name, neuron in neurons.items():
            inputs = self.inputs(name)
            neuron_data = data[name]

            if inputs:
                # if neuron has inputs
                for idx, input in enumerate(inputs):
                    # keeps pre, post order as for networkx
                    w_ij = synapses[(input, name)].sample()
                    # get latest latent value for input
                    x_j = self._latent_X[input]

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
            else:
                # if neuron has no inputs just use the mean as the prediction
                y_pred = self._latent_X[name]

            # sample from neuron
            y_pred = neuron.sample(y_pred)
            # use latest sample as latent to use for other neurons
            self._latent_X[name] = y_pred.detach()

            # conditional sample
            if neuron_data is not None and neuron.vary:
                neuron_data.sample(y_pred)
