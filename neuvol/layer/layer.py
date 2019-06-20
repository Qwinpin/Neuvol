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
from keras.layers import (Bidirectional, Conv1D, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPool1D, MaxPool2D, Reshape)
from keras.layers.recurrent import LSTM
import numpy as np

from ..constants import LAYERS_POOL, SPECIAL
from ..probabilty_pool import Distribution
from ..utils import dump


def Layer(layer_type=None, previous_layer=None, next_layer=None, **kwargs):
    if layer_type in LAYERS_MAP:
        return LAYERS_MAP[layer_type](layer_type=layer_type, previous_layer=previous_layer, next_layer=next_layer, **kwargs)
    else:
        raise TypeError()


class LayerBase:
    """
    Single layer class with compatibility checking
    """

    def __init__(self, layer_type=None, previous_layer=None, next_layer=None, **kwargs):
        self.config = {}
        self.type = layer_type
        self.options = kwargs
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        if layer_type is not None:
            self._init_parameters()
            self._check_compatibility()

    def __call__(self, net):
        layer_instance = self.init_layer()
        new_net = layer_instance(net)

        return new_net

    def _init_parameters(self):
        variables = list(LAYERS_POOL[self.type])
        for parameter in variables:
            self.config[parameter] = Distribution.layer_parameters(self.type, parameter)

    def _check_compatibility(self):
        pass

    def calculate_shape(self, previous_layer):
        """
        Shape calculator for the output

        Arguments:
            previous_layer {[type]} -- [description]
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
        serial['type'] = self.type
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
        variables = list(SPECIAL[self.type])
        for parameter in variables:
            self.config[parameter] = Distribution.layer_parameters(self.type, parameter)


class LayerComplex(LayerBase):
    def __init__(self, layer_type=None, previous_layer=None, next_layer=None, **kwargs):
        raise NotImplementedError


class LayerLSTM(LayerBase):
    def _check_compatibility(self):
        if self.next_layer is not None and self.next_layer != 'last_dense':
            self.config['return_sequences'] = True
        else:
            self.config['return_sequences'] = False

    def init_layer(self):
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
    def init_layer(self):
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
    def init_layer(self):
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
    def init_layer(self):
        layer_tf = Conv2D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerMaxPool1D(LayerBase):
    def init_layer(self):
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
    def init_layer(self):
        layer_tf = MaxPool2D(
            pool_size=[self.config['pool_size'], self.config['pool_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'])

        return layer_tf


class LayerDense(LayerBase):
    def _init_parameters(self):
        if self.type == 'last_dense':
            variables = list(LAYERS_POOL['dense'])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters('dense', parameter)
        else:
            super()._init_parameters()

    def init_layer(self):
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
        self.config['rank'] = len(self.options['shape']) + 1

    def init_layer(self):
        layer_tf = Input(
            shape=self.config['shape'])

        return layer_tf


class LayerEmbedding(LayerSpecialBase):
    def _init_parameters(self):
        super()._init_parameters()
        self.config['sentences_length'] = self.options['shape'][0]

    def init_layer(self):
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
    def init_layer(self):
        layer_tf = Flatten()

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (None, np.prod(previous_shape[1:]))

        return shape


class LayerConcat(LayerSpecialBase):
    def init_layer(self):
        layer_tf = None

        return layer_tf


class LayerReshape(LayerSpecialBase):
    def init_layer(self):
        layer_tf = Reshape(
            target_shape=self.config['target_shape'],)

        return layer_tf


class LayerDropout(LayerBase):
    def init_layer(self):
        layer_tf = Dropout(rate=self.config['rate'])

        return layer_tf


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
    'dropout': LayerDropout
}
