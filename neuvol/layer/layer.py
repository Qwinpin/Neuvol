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
from keras.layers import (Bidirectional, Conv1D, Dense, Dropout,
                          Embedding, Flatten, Input)
from keras.layers.recurrent import LSTM
import numpy as np

from ..constants import LAYERS_POOL, SPECIAL


class Layer():
    """
    Single layer class with compatibility checking
    """
    def __init__(self, layer_type, previous_layer=None, next_layer=None, **kwargs):
        self.config = {}
        self.type = layer_type
        self.options = kwargs

        self._init_parameters()
        self._check_compatibility(previous_layer, next_layer)

    def _init_parameters(self):
        if self.type == 'input':
            # set shape of the input - shape of the data
            self.config['shape'] = self.options['shape']

        elif self.type == 'embedding':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = np.random.choice(SPECIAL[self.type][parameter])

            # select the first element in the shape tuple
            self.config['sentences_length'] = self.options['shape'][0]

        elif self.type == 'last_dense':
            variables = list(LAYERS_POOL['dense'])
            for parameter in variables:
                self.config[parameter] = np.random.choice(LAYERS_POOL['dense'][parameter])

        elif self.type == 'flatten':
            pass

        else:
            variables = list(LAYERS_POOL[self.type])
            for parameter in variables:
                self.config[parameter] = np.random.choice(LAYERS_POOL[self.type][parameter])

    def _check_compatibility(self, previous_layer, next_layer):
        """
        Check data shape in specific case such as lstm or bi-lstm
        """
        if self.type == 'lstm':
            if next_layer is not None and next_layer != 'last_dense':
                self.config['return_sequences'] = True
            else:
                self.config['return_sequences'] = False

        elif self.type == 'bi':
            if next_layer is not None and next_layer != 'last_dense':
                self.config['return_sequences'] = True
            else:
                self.config['return_sequences'] = False

        elif self.type == 'last_dense':
            self.config['units'] = self.options['classes']

        elif self.type == 'cnn':
            if self.config['padding'] == 'causal':
                self.config['strides'] = 1
                if self.config['dilation_rate'] == 1:
                    self.config['padding'] = 'same'
            else:
                self.config['dilation_rate'] = 1


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

    return layer_tf
