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

from .individ_base import IndividBase
from ..layer.block import Block
from ..probabilty_pool import Distribution


class IndividText(IndividBase):
    """
    Invidiv class for text data types
    """
    def __init__(self, stage, task_type='classification', parents=None, freeze=None, **kwargs):
        IndividBase.__init__(self, stage=stage, task_type=task_type, parents=parents, freeze=freeze, **kwargs)
        self._data_processing_type = 'text'

    def _random_init_architecture(self):
        """
        At first, we set probabilities pool and the we change
        this uniform distribution according to previous layer
        """
        if self._architecture:
            self._architecture = []

        architecture = []

        # choose number of layers
        self._layers_number = Distribution.layers_number()

        # layers around current one
        previous_layer = None
        next_layer = None
        tmp_architecture = []

        # generate architecture
        for i in range(self._layers_number):
            if i != 0:
                previous_layer = architecture[i - 1].type

            if i < len(architecture) - 1:
                next_layer = architecture[i + 1].type

            if i == len(tmp_architecture) - 1:
                if self._task_type == 'classification':
                    next_layer = 'last_dense'

            # choose the number of layers in one block (like inception)

            layer = Distribution.layer()
            layers_in_block_number = np.random.choice(range(1, 5), p=[0.7, 0.1, 0.1, 0.1])

            block = Block(layer, layers_in_block_number, previous_layer, next_layer, **self.options)
            architecture.append(block)

        # Push embedding for texts
        block = Block('embedding', layers_number=1, **self.options)
        architecture.insert(0, block)

        # Push input layer for functional keras api
        block = Block('input', layers_number=1, **self.options)
        architecture.insert(0, block)

        if self._task_type == 'classification':
            # Add last layer according to task type (usually perceptron)
            block = Block('last_dense', layers_number=1, **self.options)
            architecture.append(block)
        else:
            raise TypeError('{} value not supported'.format(self._task_type))

        return architecture

    def _random_init_data_processing(self):
        if not self._architecture:
            raise Exception('Not initialized yet')

        data_tmp = {}
        data_tmp['vocabular'] = self._architecture[1].config['vocabular']
        data_tmp['sentences_length'] = self.options.get('shape', [10])[0]
        data_tmp['classes'] = self.options.get('classes', 2)

        return data_tmp
