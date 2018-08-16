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
from copy import deepcopy
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import adam, RMSprop
import numpy as np

from ..constants import EVENT, FAKE, TRAINING
from ..layer.block import Block
from ..layer.layer import init_layer
from ..probabilty_pool import Distribution


class IndividBase():
    """
    Invidiv class
    """
    # TODO: add support for different data types
    # TODO: add support for different task types

    def __init__(self, stage, task_type='classification', parents=None, freeze=None, **kwargs):
        """
        Create individ randomly or with its parents
        parents: set of two individ objects
        kwargs: dictionary with manualy specified parameters like number of classes,
        training parameters, etc
        """
        self._stage = stage
        self._data_processing_type = None
        self._task_type = task_type
        # TODO: freeze training or data parameters of individ and set manualy
        self._freeze = freeze
        self._parents = parents
        self.options = kwargs
        self._history = []
        self._name = FAKE.name().replace(' ', '_') + '_' + str(stage)
        self._architecture = []
        self._data_processing = None
        self._training_parameters = None
        self.shape_structure = None
        self._layers_number = 0
        self._result = 0.0

        if self._parents is None:
            self._random_init()
            self._history.append(EVENT('Init', stage))
        else:
            self._task_type = parents[0].task_type
            self._data_processing_type = parents[0].data_type
            self._history.append(EVENT('Birth', self._stage))

    def __str__(self):
        return self.name

    def _random_init(self):
        self._architecture = self._random_init_architecture()
        self._data_processing = self._random_init_data_processing()
        self._training_parameters = self._random_init_training()

    def _random_init_architecture(self):
        pass

    def _random_init_training(self):
        """
        Initialize training parameters
        """
        if not self._architecture:
            self._architecture = self._random_init_architecture()

        variables = list(TRAINING)
        training_tmp = {}
        for i in variables:
            training_tmp[i] = Distribution.training_parameters(i)

        return training_tmp

    def _random_init_data_processing(self):
        """
        Initialize data processing parameters
        """
        pass

    def _check_compatibility(self):
        """
        Check shapes compatibilities of different layers, modify layer if it is necessary
        """
        previous_shape = []
        shape_structure = []
        tmp = deepcopy(self._architecture)
        # use shift to know where to put additional layers
        shift = 0

        # create structure of flow shape
        for index, block in enumerate(tmp):
            # select only one layer from the block
            # we assume, that their output shape is the same
            index += shift

            if block.type == 'input':
                output_shape = block.config['shape']
            if block.type == 'embedding':
                output_shape = (2, block.config['sentences_length'], block.config['embedding_dim'])

            if block.type == 'cnn' or block.type == 'cnn2':
                filters = block.config['filters']
                kernel_size = [block.config['kernel_size']]
                padding = block.config['padding']
                strides = block.config['strides']
                dilation_rate = block.config['dilation_rate']
                input_layer = previous_shape[1:-1]
                out = []
                # convolution output shape depends on padding and stride
                if padding == 'valid':
                    if strides == 1:
                        for i, side in enumerate(input_layer):
                            out.append(side - kernel_size[i] + 1)
                    else:
                        for i, side in enumerate(input_layer):
                            out.append((side - kernel_size[i]) // strides + 1)

                elif padding == 'same':
                    if strides == 1:
                        for i, side in enumerate(input_layer):
                            out.append(side - kernel_size[i] + (2 * (kernel_size[i] // 2)) + 1)
                    else:
                        for i, side in enumerate(input_layer):
                            out.append((side - kernel_size[i] + (2 * (kernel_size[i] // 2))) // strides + 1)

                elif padding == 'causal':
                    for i, side in enumerate(input_layer):
                        out.append((side + (2 * (kernel_size[i] // 2)) - kernel_size[i] - (kernel_size[i] - 1) * (
                            dilation_rate - 1)) // strides + 1)

                # check for negative values
                if any(side <= 0 for size in out):
                    for layer in block:
                        layer.config['padding'] = 'same'

                output_shape = []
                output_shape.append(previous_shape[0])

                # *out does not work with python < 3.5
                output_shape.extend(out)

                output_shape.append(filters)
                output_shape = tuple(output_shape)

            elif block.type == 'lstm' or block.type == 'bi':
                units = block.config['units']

                # if we return sequence, output has 3-dim
                sequences = block.config['return_sequences']

                # bidirectional lstm returns double basic lstm output
                bi = 2 if block.type == 'bi' else 1

                if sequences:
                    output_shape = []
                    output_shape.append(previous_shape[0])

                    # *previous_shape[1:-1] does not work with python < 3.5
                    output_shape.extend(previous_shape[1:-1])

                    output_shape.append(units * bi)
                    output_shape = tuple(output_shape)
                else:
                    output_shape = (1, units * bi)

            elif block.type == 'dense' or block.type == 'last_dense':
                units = block.config['units']
                output_shape = []
                output_shape.append(previous_shape[0])

                # *previous_shape[1:-1] does not work with python < 3.5
                output_shape.append(previous_shape[1:-1])

                output_shape.append(units)
                output_shape = tuple(output_shape)

            elif block.type == 'flatten':
                output_shape = (1, np.prod(previous_shape[1:]))

            previous_shape = output_shape
            shape_structure.append(output_shape)

        if self._task_type == 'classification':
            # Reshape data flow in case of dimensional incompatibility
            # output shape for classifier must be 2-dim
            if shape_structure[-1][0] != 1:
                new_layer = Block('flatten', previous_block=None, next_block=None, layers_number=1)
                self._architecture.insert(-1, new_layer)

        self.shape_structure = shape_structure

    def init_tf_graph(self):
        """
        Return tensorflow graph, configurated optimizer and loss type of this individ
        """
        if not self._architecture:
            raise Exception('Non initialized net')

        network_graph_input = init_layer(self._architecture[0])
        network_graph = network_graph_input
        try:
            self._check_compatibility()
        except Exception:
            return None, None, None

        for block in self._architecture[1:]:
            if block.shape > 1:
                # we need to create list of layers and concatenate them
                tmp_block = []
                for layer in block.layers:
                    tmp_block.append(init_layer(layer)(network_graph))

                network_graph = concatenate(tmp_block, axis=-1)

            else:
                # we need just to add new layer

                network_graph = init_layer(block)(network_graph)
            # except ValueError:
            # in some cases shape of previous output could be less than kernel size of cnn
            # it leads to a negative dimension size error
            # add same padding to avoid this problem
            #    layer.config['padding'] = 'same'
            #    network_graph.add(init_layer(layer))
        model = Model(inputs=[network_graph_input], outputs=[network_graph])

        if self._training_parameters['optimizer'] == 'adam':
            optimizer = adam(
                lr=self._training_parameters['optimizer_lr'],
                decay=self._training_parameters['optimizer_decay'])
        else:
            optimizer = RMSprop(
                lr=self._training_parameters['optimizer_lr'],
                decay=self._training_parameters['optimizer_decay'])

        if self._task_type == 'classification':
            if self.options['classes'] == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'
        else:
            raise TypeError('{} value not supported'.format(self._task_type))

        return model, optimizer, loss

    @property
    def layers_number(self):
        """
        Get the number of layers in the network
        """
        return self._layers_number

    @property
    def data_type(self):
        """
        Get the type of data
        """
        return self._data_processing_type

    @property
    def task_type(self):
        """
        Get the type of task
        """
        return self._task_type

    @property
    def history(self):
        """
        Get the history of this individ
        """
        return self._history

    @property
    def classes(self):
        """
        Get the number of classes if it is exists, None otherwise
        """
        if self.options.get("classes", None) is not None:
            return self.options['classes']
        else:
            return None

    @property
    def name(self):
        """
        Get the name of this individ
        """
        return self._name

    @property
    def stage(self):
        """
        Get the stage of birth
        """
        return self._stage

    @property
    def architecture(self):
        """
        Get the architecture in pure form: list of Block's object
        """
        return self._architecture

    @property
    def parents(self):
        """
        Get parents of this individ
        """
        return self._parents

    @property
    def schema(self):
        """
        Get the network schema in textual form
        """
        schema = [(i.type, i.config_all) for i in self._architecture]

        return schema

    @property
    def data_processing(self):
        """
        Get the data processing parameters
        """
        return self._data_processing

    @property
    def training_parameters(self):
        """
        Get the training parameters
        """
        return self._training_parameters

    @property
    def result(self):
        """
        Get the result of the efficiency (f1 or AUC)
        """
        return self._result

    @result.setter
    def result(self, value):
        """
        Set new fitness result
        """
        self._result = value

    @history.setter
    def history(self, event):
        """
        Add new event to the history
        """
        self._history.append(event)

    def random_init(self):
        """
        Public method for calling the random initialisation
        """
        self._random_init()

    def random_init_architecture(self):
        """
        Public method for calling the random architecture initialisation
        """
        return self._random_init_architecture()

    def random_init_data_processing(self):
        """
        Public method for calling the random data processing initialisation
        """
        return self._random_init_data_processing()

    def random_init_training(self):
        """
        Public method for calling the random training initialisation
        """
        return self._random_init_training()

    @data_processing.setter
    def data_processing(self, data_processing):
        """
        Set a new data processing config
        """
        self._data_processing = data_processing

    @training_parameters.setter
    def training_parameters(self, training_parameters):
        """
        Set a new training parameters
        """
        self._training_parameters = training_parameters

    @architecture.setter
    def architecture(self, architecture):
        """
        Set a new architecture
        """
        self._architecture = architecture
