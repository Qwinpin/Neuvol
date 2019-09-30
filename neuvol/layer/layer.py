# Copyright 2018 Timur Sokhin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras.layers import (Bidirectional, concatenate, Conv1D, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPool1D, MaxPool2D, Reshape, RepeatVector,
                          SeparableConv1D, SeparableConv2D, Conv2DTranspose)
from keras.layers.recurrent import LSTM
import numpy as np

from ..constants import LAYERS_POOL, SPECIAL
from ..probabilty_pool import Distribution
from ..utils import dump


def Layer(layer_type, options=None, previous_layer=None, next_layer=None):
    """
    Factory for the Layers instances
    """
    if layer_type in LAYERS_MAP:
        return LAYERS_MAP[layer_type](layer_type=layer_type, previous_layer=previous_layer, next_layer=next_layer, options=options)
    else:
        raise TypeError()


class LayerBase:
    """
    Single layer class with compatibility checking
    In order to keep logical connectivity with Keras layer instances
    __call__ method is implemented.
    Then calling ranks comparison is performed - in case of incompatibilities
    of rank add reshape layer.
    """

    def __init__(self, layer_type, previous_layer=None, next_layer=None, options=None):
        self.config = {}
        self.layer_type = layer_type
        self.options = options
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        if layer_type is not None:
            self._init_parameters()
            self._check_compatibility()

    def __call__(self, net, previous_layer):
        """
        Add layer to a network tail, previous layer is required for shape and rank check
        In case of multiple layers concatenation layer is injected
        """
        # in case of concatenation
        if isinstance(net, list):
            concat_layer = Layer('concat')
            net = concat_layer(net, previous_layer)
            previous_layer = concat_layer

        reshape_layer = self._init_reshape_layer(previous_layer)

        if reshape_layer is None:
            self.config['rank'] = self.calculate_rank(previous_layer)
            self.config['shape'] = self.calculate_shape(previous_layer)
        else:
            self.config['rank'] = self.calculate_rank(reshape_layer)
            self.config['shape'] = self.calculate_shape(reshape_layer)

        if reshape_layer is not None:
            new_new = reshape_layer(net, previous_layer)
        else:
            new_new = net

        layer_instance = self.init_layer(previous_layer)
        new_net = layer_instance(new_new)

        return new_net

    def init_layer(self, previous_layer):
        """
        Each layer type has its own initialization
        """
        return ...

    def _init_reshape_layer(self, previous_layer):
        """
        Add reshape layer if ranks is different
        """
        if self.config['input_rank'] != previous_layer.rank:
            reshape_layer = reshaper(previous_layer, self)
        else:
            reshape_layer = None

        return reshape_layer

    def _init_parameters(self):
        """
        Get random values of all required parameters
        """
        variables = list(LAYERS_POOL[self.layer_type])
        for parameter in variables:
            self.config[parameter] = Distribution.layer_parameters(self.layer_type, parameter)

        if self.options is not None and self.options.get('input_rank') is not None:
            self.config['input_rank'] = self.options['input_rank']

    def _check_compatibility(self):
        pass

    def calculate_shape(self, previous_layer):
        """
        Shape calculator for the output

        Arguments:
            previous_layer {Layer} -- previous layer, which is connected to the current
        """
        previous_shape = previous_layer.shape
        new_shape = previous_shape

        return new_shape

    def calculate_rank(self, previous_layer):
        previous_rank = previous_layer.rank
        new_rank = previous_rank

        return new_rank

    def save(self):
        """
        Serialization of layer
        """
        serial = dict()
        serial['config'] = self.config
        serial['type'] = self.layer_type
        serial['options'] = self.options
        serial['previous_layer'] = self.previous_layer
        serial['next_layer'] = self.next_layer

        return serial

    def dump(self, path):
        """
        Dump layer info
        """
        dump(self.save(), path)

    @property
    def shape(self):
        return self.config['shape']

    @property
    def rank(self):
        return self.config['rank']

    @shape.setter
    def shape(self, value):
        self.config['shape'] = value

    @rank.setter
    def rank(self, value):
        self.config['shape'] = value

    @staticmethod
    def load(serial):
        """
        Deserialization of layer
        """
        layer = LayerBase(None)
        layer.config = serial['config']
        layer.type = serial['type']
        layer.options = serial['options']
        layer.previous_layer = serial['previous_layer']
        layer.next_layer = serial['next_layer']

        return layer


class LayerSpecialBase(LayerBase):
    def _init_parameters(self):
        variables = list(SPECIAL[self.layer_type])
        for parameter in variables:
            self.config[parameter] = Distribution.layer_parameters(self.layer_type, parameter)


class LayerComplex(LayerBase):
    def __init__(self, layer_type=None, previous_layer=None, next_layer=None, options=None):
        raise NotImplementedError


class LayerLSTM(LayerBase):
    def _check_compatibility(self):
        if self.next_layer is not None and self.next_layer != 'last_dense':
            self.config['return_sequences'] = True
        else:
            self.config['return_sequences'] = False

    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = LSTM(
            units=self.config['units'],
            recurrent_dropout=self.config['recurrent_dropout'],
            activation=self.config['activation'],
            implementation=self.config['implementation'],
            return_sequences=self.config['return_sequences'])

        return layer_tf

    def calculate_rank(self, previous_layer):
        if self.config['return_sequences']:
            rank = 3
        else:
            rank = 2

        return rank

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape

        if self.config['return_sequences']:
            shape = (None, previous_shape[1:-1], self.config['units'])
        else:
            shape = (None, self.config['units'])

        return shape


class LayerBiLSTM(LayerLSTM):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Bidirectional(
            LSTM(
                units=self.config['units'],
                recurrent_dropout=self.config['recurrent_dropout'],
                activation=self.config['activation'],
                implementation=self.config['implementation'],
                return_sequences=self.config['return_sequences']))

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape

        if self.config['return_sequences']:
            shape = (None, previous_shape[1:-1], 2*self.config['units'])
        else:
            shape = (None, 2*self.config['units'])

        return shape


class LayerCNN1D(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Conv1D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size']],
            strides=[self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf

    def _check_compatibility(self):
        super()._check_compatibility()
        if self.config['dilation_rate'] > 1:
            self.config['strides'] = 1

        elif self.config['dilation_rate'] == 1 and self.config['padding'] == 'causal':
            self.config['padding'] = 'same'

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        filters = self.config['filters']
        kernel_size = self.config['kernel_size']

        if kernel_size % 2 == 0:
            align = 1
        else:
            align = 0

        padding = self.config['padding']
        strides = self.config['strides']
        dilation_rate = self.config['dilation_rate']

        if padding == 'valid':
            if dilation_rate != 1:
                out = [(i - kernel_size - (kernel_size - 1) * (dilation_rate - 1)) // strides + 1 - align
                       for i in previous_shape[1:-1]]
            else:
                out = [((i - kernel_size) // strides + 1 - align) for i in previous_shape[1:-1]]

        elif padding == 'same':
            out = [((i - kernel_size + (2 * (kernel_size // 2))) // strides + 1 - align) for i in previous_shape[1:-1]]

        elif padding == 'causal':
            out = [i for i in previous_shape[1:-1]]
            # out = [(i - kernel_size - (kernel_size - 1) * (dilation_rate - 1)) // strides + 1 - align
            #        for i in previous_shape[1:-1]]

        for i in out:
            # if some of the layer too small - change the padding
            if i <= 0:
                self.config['padding'] = 'same'
                shape = self.calculate_shape(previous_layer)
                return shape

        shape = (None, *out, filters)

        return shape


class LayerCNN2D(LayerCNN1D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Conv2D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerMaxPool1D(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = MaxPool1D(
            pool_size=[self.config['pool_size']],
            strides=[self.config['strides']],
            padding=self.config['padding'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape

        kernel_size = self.config['pool_size']
        strides = self.config['strides']
        padding = self.config['padding']
        if kernel_size % 2 == 0:
            align = 1
        else:
            align = 0

        if padding == 'same':
            out = [((i + 2*(kernel_size // 2) - kernel_size) // strides + 1 - align) for i in previous_shape[1:-1]]
        else:
            out = [((i - kernel_size) // strides + 1 - align) for i in previous_shape[1:-1]]

        for i in out:
            # if some of the layer too small - change the padding
            if i <= 0:
                self.config['padding'] = 'same'
                shape = self.calculate_shape(previous_layer)
                return shape

        shape = (None, *out, previous_shape[-1])

        return shape


class LayerMaxPool2D(LayerMaxPool1D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = MaxPool2D(
            pool_size=[self.config['pool_size'], self.config['pool_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'])

        return layer_tf


class LayerDense(LayerBase):
    def _init_parameters(self):
        if self.layer_type == 'last_dense':
            variables = list(LAYERS_POOL['dense'])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters('dense', parameter)
        else:
            super()._init_parameters()

    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Dense(
            units=self.config['units'],
            activation=self.config['activation'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (*previous_shape[:-1], self.config['units'])

        return shape


class LayerInput(LayerBase):
    def _init_parameters(self):
        self.config['shape'] = self.options['shape']
        self.config['rank'] = len(self.options['shape'])

    def init_layer(self, previous_layer):
        layer_tf = Input(
            shape=self.config['shape'][1:])  # all except first None

        return layer_tf


class LayerEmbedding(LayerSpecialBase):
    def _init_parameters(self):
        super()._init_parameters()
        self.config['sentences_length'] = self.options['shape'][0]

    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Embedding(
            input_dim=self.config['vocabular'],
            output_dim=self.config['embedding_dim'],
            input_length=self.config['sentences_length'],
            trainable=self.config['trainable'])

        return layer_tf

    def calculate_rank(self, previous_layer):
        rank = 3

        return rank

    def calculate_shape(self, previous_layer):
        shape = (None, self.config['sentences_length'], self.config['embedding_dim'])

        return shape


class LayerFlatten(LayerSpecialBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Flatten()

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (None, np.prod(previous_shape[1:]))

        return shape


class LayerConcat(LayerSpecialBase):
    def __call__(self, nets, previous_layers):
        reshape_layers = self.merger_mass(previous_layers)

        new_nets = []
        for i, reshape_layer in enumerate(reshape_layers):
            new_nets.append(reshape_layer(nets[i], previous_layers[i]))
        # new_nets = [reshape_layer(nets[i], previous_layers[i]) for i, reshape_layer in enumerate(reshape_layers)]
        new_net = concatenate(new_nets)

        return new_net

    def merger_mass(self, layers):
        shape_modifiers = []
        for layer in layers:
            new_shape = (None, np.prod(layer.config['shape'][1:]))

            reshape = Layer('reshape')
            reshape.config['target_shape'] = new_shape[1:]
            reshape.config['shape'] = new_shape
            reshape.config['input_rank'] = layer.config['rank']
            reshape.config['rank'] = 2

            shape_modifiers.append(reshape)

        shapes = [shape_modifiers[i].config['shape'][1:] for i, layer in enumerate(layers)]
        self.config['shape'] = (None, np.sum(shapes))
        self.config['rank'] = 2

        return shape_modifiers


class LayerReshape(LayerSpecialBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Reshape(
            target_shape=self.config['target_shape'],)

        return layer_tf


class LayerDropout(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Dropout(rate=self.config['rate'])

        return layer_tf


class LayerRepeatVector(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = RepeatVector(n=self.config['n'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (None, self.config['n'], *previous_shape[1:])

        return shape

    def calculate_rank(self, previous_layer):
        previous_rank = previous_layer.rank
        rank = previous_rank + 1

        return rank


class LayerSepCNN1D(LayerCNN1D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = SeparableConv1D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size']],
            strides=[self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerSepCNN2D(LayerCNN2D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = SeparableConv2D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerDeCNN2D(LayerCNN2D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Conv2DTranspose(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            output_padding=self.config['output_padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        filters = self.config['filters']
        kernel_size = self.config['kernel_size']

        padding = self.config['padding']
        output_padding = self.config['output_padding']
        strides = self.config['strides']
        dilation_rate = self.config['dilation_rate']

        if padding == 'valid':
            if output_padding is None:
                output_padding = 0
            out = [(i - 1) * strides + kernel_size + (kernel_size - 1) * (dilation_rate - 1) + output_padding for i in previous_shape[1:-1]]

        elif padding == 'same':
            if output_padding is None:
                out = [i * strides for i in previous_shape[1:-1]]
            else:
                out = [(i - 1) * strides + kernel_size - 2 * (kernel_size // 2) + output_padding for i in previous_shape[1:-1]]

        shape = (None, *out, filters)

        return shape


LAYERS_MAP = {
    'input': LayerInput,
    'lstm': LayerLSTM,
    'bi': LayerBiLSTM,
    'cnn': LayerCNN1D,
    'cnn2': LayerCNN2D,
    'max_pool': LayerMaxPool1D,
    'max_pool2': LayerMaxPool2D,
    'dense': LayerDense,
    'embedding': LayerEmbedding,
    'flatten': LayerFlatten,
    'concat': LayerConcat,
    'reshape': LayerReshape,
    'dropout': LayerDropout,
    'repeatvector': LayerRepeatVector,
    'separablecnn': LayerSepCNN1D,
    'separablecnn2': LayerSepCNN2D,
    'decnn2': LayerDeCNN2D
}


def reshaper_shape(difference, prev_layer):
    if difference > 0:
        # new shape is a dimensions flatten together, if we need to move from rank 5 to 2
        # we should take all dim from 1 (first - batch size, always None) and till the
        # difference + 1
        # (None, 12, 124, 124, 10) -> rank 2 -> (None, 12*124*124*10)
        # (None, 12, 124, 124, 10) -> rank 4 -> (None, 12, 124*124, 10)
        prev_rank = prev_layer.rank

        # how many elements will be multiplied
        window = difference + 1
        # how many elements will not be multiplied
        # -1 used because None elements is not taken into account
        free_elements = prev_rank - window - 1

        if free_elements == 0:
            tail_ind = None
            front_ind = 1

        else:
            tail_ind = -1
            front_ind = free_elements

        front = prev_layer.config['shape'][1: front_ind]
        mid = np.prod(prev_layer.config['shape'][front_ind: tail_ind])

        if tail_ind is None:
            tail = []
        else:
            tail = prev_layer.config['shape'][-1:]

        new_shape = (
            None,
            *front,
            mid,
            *tail)

    elif difference < 0:
        # simple reshape with 1 dims
        # add new dims to remove the difference
        new_shape = (
            None,
            *prev_layer.config['shape'][1:],
            *([1] * abs(difference)))

    else:
        new_shape = prev_layer.config['shape']
    return new_shape


def reshaper(prev_layer, layer):
    """
    Restore compability between layer with diff ranks
    """
    if layer.config['input_rank'] is None:
        difference = 0
    else:
        difference = (prev_layer.config['rank'] - layer.config['input_rank'])

    if difference == 0:
        return None

    new_shape = reshaper_shape(difference, prev_layer)

    modifier = Layer('reshape')
    modifier.config['target_shape'] = new_shape[1:]
    modifier.config['shape'] = new_shape
    modifier.config['input_rank'] = prev_layer.config['rank']
    modifier.config['rank'] = layer.config['input_rank']

    return modifier
