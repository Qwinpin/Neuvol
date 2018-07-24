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
from keras.models import Sequential
from keras.optimizers import RMSprop, adam

from ..constants import TRAINING, EVENT, FAKE
from ..layer import Layer, init_layer


class IndividBase():
    """
    Invidiv class
    """
    # TODO: add support for different data types
    # TODO: add support for different task types
    # TODO: all data types as subclass?
    # TODO: network parser to avoid layer compatibility errors and shape errors

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
        self._history = [EVENT('Init', stage)]
        self._name = FAKE.name().replace(' ', '_') + '_' + str(stage)
        self._architecture = []
        self._data_processing = None
        self._training_parameters = None
        self.shape_structure = None
        self._layers_number = 0
        self._result = 0.0

        if self._parents is None:
            self._random_init()
        else:
            self._init_with_crossing()

        self._check_compatibility()

    def __str__(self):
        return None

    def _random_init(self):
        self._architecture = self._random_init_architecture()
        self._data_processing = self._random_init_data_processing()
        self._training_parameters = self._random_init_training()

    def _random_init_architecture(self):
        pass

    def _init_with_crossing(self):
        """
        New individ parameters according its parents (only 2 now, classic)
        """
        # TODO: add compatibility checker after all crossing
        father = self._parents[0]
        mother = self._parents[1]
        # father_architecture - chose architecture from first individ and text
        # and train from second
        # father_training - only training config from first one
        # father_arch_layers - select overlapping layers
        # and replace parameters from the first architecture
        # with parameters from the second

        pairing_type = np.random.choice([
            'father_architecture',
            'father_training',
            'father_architecture_layers',
            'father_architecture_parameter',
            'father_data_processing'])

        self._history.append(EVENT('Birth', self._stage))

        if pairing_type == 'father_architecture':
            # Father's architecture and mother's training and data
            self._architecture = father.architecture
            self._training_parameters = mother.training_parameters
            self._data_processing = mother.data_processing

            # change data processing parameter to avoid incompatibility
            self._data_processing['sentences_length'] = father.data_processing['sentences_length']

        elif pairing_type == 'father_training':
            # Father's training and mother's architecture and data
            self._architecture = mother.architecture
            self._training_parameters = father.training_parameters
            self._data_processing = mother.data_processing

        elif pairing_type == 'father_architecture_layers':
            # Select father's architecture and replace random layer with mother's layer
            changes_layer = np.random.choice([i for i in range(1, len(self._architecture))])
            alter_layer = np.random.choice([i for i in range(1, len(mother.architecture))])

            self._architecture = father.architecture
            self._architecture[changes_layer] = mother.architecture[alter_layer]
            self._training_parameters = father.training_parameters
            self._data_processing = father.data_processing

        elif pairing_type == 'father_architecture_parameter':
            # Select father's architecture and change layer parameters with mother's layer
            # dont touch first and last elements - embedding and dense(3),
            # too many dependencies with text model
            # select common layer
            tmp_father = [layer.type for layer in father.architecture[1:-1]]
            tmp_mother = [layer.type for layer in mother.architecture[1:-1]]

            intersections = set(tmp_father) & set(tmp_mother)

            if not intersections:
                self._architecture = father.architecture
                self._training_parameters = father.training_parameters
                self._data_processing = father.data_processing

            intersected_layer = np.random.choice(list(intersections))

            # add 1, because we did not take into account first layer
            changes_layer = tmp_father.index(intersected_layer) + 1
            alter_layer = tmp_mother.index(intersected_layer) + 1

            self._architecture = father.architecture
            self._architecture[changes_layer] = mother.architecture[alter_layer]
            self._training_parameters = father.training_parameters
            self._data_processing = father.data_processing

        elif pairing_type == 'father_data_processing':
            # Select father's data processing and mother's architecture and training
            # change mother's embedding to avoid mismatchs in dimensions
            self._architecture = mother.architecture
            self._training_parameters = mother.training_parameters
            self._data_processing = father.data_processing

            # change data processing parameter to avoid incompatibility
            self._architecture[0] = father.architecture[0]

    def _random_init_training(self):
        """
        Initialize training parameters
        """
        if not self._architecture:
            raise Exception('Not initialized yet')

        variables = list(TRAINING)
        training_tmp = {}
        for i in variables:
            training_tmp[i] = np.random.choice(TRAINING[i])

        return training_tmp

    def _random_init_data_processing(self):
        """
        Initialize data processing parameters
        """
        pass

    def _check_compatibility(self):
        """
        Check shapes compatibilities, modify layer if it is necessary
        """
        previous_shape = []
        shape_structure = []
        # create structure of flow shape
        for layer in self._architecture:
            if layer.type == 'embedding':
                output_shape = (2, layer.config['sentences_length'], layer.config['embedding_dim'])

            if layer.type == 'cnn':
                filters = layer.config['filters']
                kernel_size = [layer.config['kernel_size']]
                padding = layer.config['padding']
                strides = layer.config['strides']
                dilation_rate = layer.config['dilation_rate']
                input = previous_shape[1:-1]
                out = []

                # convolution output shape depends on padding and stride
                if padding == 'valid':
                    if strides == 1:
                        for i, side in enumerate(input):
                            out.append(side - kernel_size[i] + 1)
                    else:
                        for i, side in enumerate(input):
                            out.append((side - kernel_size[i]) // strides + 1)

                elif padding == 'same':
                    if strides == 1:
                        for i, side in enumerate(input):
                            out.append(side - kernel_size[i] + (2 * (kernel_size[i] // 2)) + 1)
                    else:
                        for i, side in enumerate(input):
                            out.append((side - kernel_size[i] + (2 * (kernel_size[i] // 2))) // strides + 1)

                elif padding == 'causal':
                    for i, side in enumerate(input):
                        out.append((side + (2 * (kernel_size[i] // 2)) - kernel_size[i] - (kernel_size[i] - 1) * (
                                    dilation_rate - 1)) // strides + 1)

                # check for negative values
                if any(side <= 0 for size in out):
                    layer.config['padding'] = 'same'
                output_shape = (previous_shape[0], *out, filters)

            elif layer.type == 'lstm' or layer.type == 'bi':
                units = layer.config['units']

                # if we return sequence, output has 3-dim
                sequences = layer.config['return_sequences']

                # bidirectional lstm returns double basic lstm output
                bi = 2 if layer.type == 'bi' else 1

                if sequences:
                    output_shape = (previous_shape[0], *previous_shape[1:-1], units * bi)
                else:
                    output_shape = (1, units * bi)

            elif layer.type == 'dense' or layer.type == 'last_dense':
                units = layer.config['units']
                output_shape = (previous_shape[0], *previous_shape[1:-1], units)

            elif layer.type == 'flatten':
                output_shape = (1, np.prod(previous_shape[1:]))

            previous_shape = output_shape
            shape_structure.append(output_shape)

        if self._task_type == 'classification':
            # Reshape data flow in case of dimensional incompatibility
            # output shape for classifier must be 2-dim
            if shape_structure[-1][0] != 1:
                new_layer = Layer('flatten', None, None)
                self._architecture.insert(-1, new_layer)

        self.shape_structure = shape_structure

    def init_tf_graph(self):
        """
        Return tensorflow graph from individ architecture
        """
        if not self._architecture:
            raise Exception('Non initialized net')

        network_graph = Sequential()
        self._check_compatibility()

        for i, layer in enumerate(self._architecture):
            try:
                network_graph.add(init_layer(layer))
            except ValueError as e:
                # in some cases shape of previous output could be less than kernel size of cnn
                # it leads to a negative dimension size error
                # add same padding to avoid this problem
                layer.config['padding'] = 'same'
                network_graph.add(init_layer(layer))

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
            raise Exception('Unsupported task type')

        return network_graph, optimizer, loss

    def crossing(self, other, stage):
        pass

    @property
    def layers_number(self):
        return self._layers_number

    @property
    def data_type(self):
        return self._data_processing_type

    @property
    def task_type(self):
        return self._task_type

    @property
    def history(self):
        return self._history

    @property
    def classes(self):
        return self.options['classes']

    @property
    def name(self):
        return self._name

    @property
    def stage(self):
        return self._stage

    @property
    def architecture(self):
        return self._architecture

    @property
    def parents(self):
        return self._parents

    @property
    def schema(self):
        """
        Return network schema
        """
        schema = [(i.type, i.config) for i in self._architecture]

        return schema

    @property
    def data_processing(self):
        """
        Return data processing parameters
        """
        return self._data_processing

    @property
    def training_parameters(self):
        """
        Return training parameters
        """
        return self._training_parameters

    @property
    def result(self):
        """
        Return fitness measure
        """
        return self._result

    @result.setter
    def result(self, value):
        """
        New fitness result
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
        Public method for random initialisation
        """
        self._random_init()

    def random_init_architecture(self):
        """
        Public method for random architecture initialisation
        """
        return self._random_init_architecture()

    def random_init_data_processing(self):
        """
        Public method for random data processing initialisation
        """
        return self._random_init_data_processing()

    def random_init_training(self):
        """
        Public method for random training initialisation
        """
        return self._random_init_training()

    @data_processing.setter
    def data_processing(self, data_processing):
        self._data_processing = data_processing

    @training_parameters.setter
    def training_parameters(self, training_parameters):
        self._training_parameters = training_parameters

    @architecture.setter
    def architecture(self, architecture):
        self._architecture = architecture
