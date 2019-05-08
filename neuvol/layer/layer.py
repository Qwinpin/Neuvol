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
                          Embedding, Flatten, Input, MaxPool1D, MaxPool2D, Reshape)
from keras.layers.recurrent import LSTM

from ..constants import LAYERS_POOL, SPECIAL
from ..probabilty_pool import Distribution
from ..utils import dump


class Layer():
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

    def _init_parameters(self):
        if self.type == 'input':
            # set shape of the input - shape of the data
            self.config['shape'] = self.options['shape']
            self.config['rank'] = self.options['rank']

        # TODO: refactor
        elif self.type == 'embedding':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters(self.type, parameter)

            # select the first element in the shape tuple
            self.config['sentences_length'] = self.options['shape'][0]

        elif self.type == 'reshape':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters(self.type, parameter)

        elif self.type == 'last_dense':
            variables = list(LAYERS_POOL['dense'])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters('dense', parameter)

        elif self.type == 'flatten':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters(self.type, parameter)

        elif self.type == 'concat':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters(self.type, parameter)

        else:
            variables = list(LAYERS_POOL[self.type])
            for parameter in variables:
                self.config[parameter] = Distribution.layer_parameters(self.type, parameter)

    def _check_compatibility(self):
        """
        Check data shape in specific case such as lstm or bi-lstm
        """
        if self.type == 'lstm':
            if self.next_layer is not None and self.next_layer != 'last_dense':
                self.config['return_sequences'] = True
            else:
                self.config['return_sequences'] = False

        elif self.type == 'bi':
            if self.next_layer is not None and self.next_layer != 'last_dense':
                self.config['return_sequences'] = True
            else:
                self.config['return_sequences'] = False

        elif self.type == 'last_dense':
            self.config['units'] = self.options['classes']

        elif self.type == 'cnn' or self.type == 'cnn2':
            # control dilation constraints
            if self.config['dilation_rate'] != 1:
                self.config['strides'] = 1

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

    @staticmethod
    def load(serial):
        """
        Deserialization of layer
        """
        layer = Layer(None)
        layer.config = serial['config']
        layer.type = serial['type']
        layer.options = serial['options']
        layer.previous_layer = serial['previous_layer']
        layer.next_layer = serial['next_layer']

        return layer


def init_layer(layer):
    """
    Return layer according its configs as keras object
    """
    if layer.type == 'input':
        layer_tf = Input(
            shape=layer.config['shape'])

    elif layer.type == 'lstm':
        layer_tf = LSTM(
            units=layer.config['units'],
            recurrent_dropout=layer.config['recurrent_dropout'],
            activation=layer.config['activation'],
            implementation=layer.config['implementation'],
            return_sequences=layer.config['return_sequences'])

    elif layer.type == 'bi':
        layer_tf = Bidirectional(
            LSTM(
                units=layer.config['units'],
                recurrent_dropout=layer.config['recurrent_dropout'],
                activation=layer.config['activation'],
                implementation=layer.config['implementation'],
                return_sequences=layer.config['return_sequences']))

    elif layer.type == 'dense':
        layer_tf = Dense(
            units=layer.config['units'],
            activation=layer.config['activation'])

    elif layer.type == 'last_dense':
        layer_tf = Dense(
            units=layer.config['units'],
            activation=layer.config['activation'])

    elif layer.type == 'cnn':
        layer_tf = Conv1D(
            filters=layer.config['filters'],
            kernel_size=[layer.config['kernel_size']],
            strides=[layer.config['strides']],
            padding=layer.config['padding'],
            dilation_rate=tuple([layer.config['dilation_rate']]),
            activation=layer.config['activation'])

    elif layer.type == 'cnn2':
        layer_tf = Conv2D(
            filters=layer.config['filters'],
            kernel_size=[layer.config['kernel_size'], layer.config['kernel_size']],
            strides=[layer.config['strides'], layer.config['strides']],
            padding=layer.config['padding'],
            dilation_rate=tuple([layer.config['dilation_rate'], layer.config['dilation_rate']]),
            activation=layer.config['activation'])

    elif layer.type == 'max_pool':
        layer_tf = MaxPool1D(
            pool_size=[layer.config['pool_size']],
            strides=[layer.config['strides']],
            padding=layer.config['padding'])

    elif layer.type == 'max_pool2':
        layer_tf = MaxPool2D(
            pool_size=[layer.config['pool_size'], layer.config['pool_size']],
            strides=[layer.config['strides'], layer.config['strides']],
            padding=layer.config['padding'])

    elif layer.type == 'dropout':
        layer_tf = Dropout(rate=layer.config['rate'])

    elif layer.type == 'embedding':
        layer_tf = Embedding(
            input_dim=layer.config['vocabular'],
            output_dim=layer.config['embedding_dim'],
            input_length=layer.config['sentences_length'],
            trainable=layer.config['trainable'])

    elif layer.type == 'flatten':
        layer_tf = Flatten()
    
    elif layer.type == 'concat':
        layer_tf = None
    
    elif layer.type == 'reshape':
        layer_tf = Reshape(
            target_shape=layer.config['target_shape'],
        )

    return layer_tf
