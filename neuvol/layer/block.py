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
from .layer import Layer
from ..utils import dump


class Block():
    """
    Block of layers class
    """
    def __init__(self, layers_type=None, layers_number=1, previous_block=None, next_block=None, **kwargs):
        self.layers = None
        self.layer_type = layers_type
        self.shape = layers_number
        self.previous_block = previous_block
        self.next_block = next_block
        self.options = kwargs

        if layers_type is not None:
            self._init_parameters()
            self._check_compatibility()

    def _init_parameters(self):
        """
        Initialize block
        """
        previous_layers = self.previous_block
        next_layer = self.next_block

        tmp_kwargs = self.options
        self.layers = [Layer(self.layer_type, previous_layers, next_layer, **tmp_kwargs) for _ in range(self.shape)]

    def _check_compatibility(self):
        """
        For the output shape compatibility we need to freeze padding as the 'same'
        """
        # TODO: layers and block compatibility checker in one place
        if self.shape > 1:
            if self.layer_type == 'dropout':
                pass

            elif self.layer_type == 'cnn' or self.layer_type == 'cnn2':
                # note that same padding and strides != 1 is inconsistent in keras
                for layer in self.layers:
                    layer.config['padding'] = 'same'
                    layer.config['strides'] = 1

            elif self.layer_type == 'max_pool' or self.layer_type == 'max_pool2':
                # note that same padding and strides != 1 is inconsistent in keras
                for layer in self.layers:
                    layer.config['padding'] = 'same'
                    layer.config['strides'] = 1

            else:
                output = self.layers[0].config['units']
                for layer in self.layers:
                    layer.config['units'] = output

    def save(self):
        """
        Serialization of block
        """
        serial = dict()
        serial['type'] = self.layer_type
        serial['shape'] = self.shape
        serial['previous_block'] = self.previous_block
        serial['next_block'] = self.next_block
        serial['options'] = self.options
        serial['layers'] = [layer.save() for layer in self.layers]

        return serial

    def dump(self, path):
        """
        Dump block info
        """
        dump(self.save(), path)

    @staticmethod
    def load(serial):
        """
        Deserialization of block
        """
        block = Block()
        block.layers = [Layer.load(layer) for layer in serial['layers']]
        block.type = serial['type']
        block.shape = serial['shape']
        block.previous_block = serial['previous_block']
        block.next_block = serial['next_block']
        block.options = serial['options']

        return block

    @property
    def config(self):
        """
        Get config of the first layer in a block
        """
        return self.layers[0].config

    @property
    def config_all(self):
        """
        Return configs of all layers
        """
        info = [layer.config for layer in self.layers]

        return info
