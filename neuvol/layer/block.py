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


class Block():
    """
    Block of layers class
    """
    def __init__(self, layers_type, layers_number=1, previous_block=None, next_block=None, **kwargs):
        self.layers = None
        self.type = layers_type
        self.shape = layers_number
        self.previous_block = previous_block
        self.next_block = next_block
        self.options = kwargs

        self._init_parameters()
        self._check_compatibility()

    def _init_parameters(self):
        """
        Initialize block
        """
        previous_layers = self.previous_block if self.previous_block is not None else None
        next_layer = self.next_block if self.next_block is not None else None

        tmp_kwargs = self.options
        self.layers = [Layer(self.type, previous_layers, next_layer, **tmp_kwargs) for _ in range(self.shape)]

    def _check_compatibility(self):
        """
        For the output shape compatibility we need to freeze padding as the 'same'
        """
        # TODO: layers and block compatibility checker in one place
        if self.shape > 1:
            if self.type == 'dropout':
                pass

            elif self.type == 'cnn':
                strides = self.layers[0].config['strides']
                dilation_rate = self.layers[0].config['dilation_rate']
                for layer in self.layers:
                    layer.config['padding'] = 'same'
                    layer.config['strides'] = strides
                    layer.config['dilation_rate'] = dilation_rate

            else:
                output = self.layers[0].config['units']
                for layer in self.layers:
                    layer.config['units'] = output

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
