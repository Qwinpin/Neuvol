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
import numpy as np

from .constants import LAYERS_POOL, SPECIAL


class Layer():
    """
    Single layer class with compatibility checking
    """

    def __init__(self, layer_type, previous_layer=None, next_layer=None, classes=None):
        self._classes = classes
        self.config = {}
        self.type = layer_type

        self._init_parameters()
        self._check_compatibility(previous_layer, next_layer)

    def _init_parameters(self):
        if self.type == 'embedding':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = np.random.choice(SPECIAL[self.type][parameter])

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
        TODO: check negative dimension size in case of convolution layers
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
            self.config['units'] = self._classes

        elif self.type == 'cnn':
            if self.config['padding'] == 'causal':
                self.config['strides'] = 1
                if self.config['dilation_rate'] == 1:
                    self.config['padding'] = 'same'
            else:
                self.config['dilation_rate'] = 1