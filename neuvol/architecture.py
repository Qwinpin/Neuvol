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

from collections import namedtuple

import faker
import numpy as np
from keras.layers import (Bidirectional, Conv1D, Dense, Dropout,
                          Embedding, Flatten)
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop, adam


FLOAT32 = np.float32

# Training parameters
TRAINING = {
    'batchs': [i for i in range(8, 512, 32)],
    'epochs': [i for i in range(1, 25) if i % 2],
    'optimizer': ['adam', 'RMSprop'],
    'optimizer_decay': [FLOAT32(i / 10000) for i in range(1, 500, 5)],
    'optimizer_lr': [FLOAT32(i / 10000) for i in range(1, 500, 5)]}

# Specific parameters
SPECIAL = {
    'embedding': {
        'vocabular': [5000, 8000, 10000, 15000, 20000, 25000, 30000],
        'sentences_length': [10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150],
        'embedding_dim': [64, 128, 200, 300],
        'trainable': [False, True]}}

LAYERS_POOL = {
    'bi': {
        'units': [1, 2, 4, 8, 12, 16],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 5)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2]},

    'lstm': {
        'units': [1, 2, 4, 8, 12, 16],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 5)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2]},

    'cnn': {
        'filters': [4, 8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7],
        'strides': [1, 2, 3],
        'padding': ['valid', 'same', 'causal'],
        'activation': ['tanh', 'relu'],
        'dilation_rate': [1, 2, 3]},

    'dense': {
        'units': [16, 64, 128, 256, 512],
        'activation': ['softmax', 'sigmoid']},

    'dropout': {'rate': [FLOAT32(i / 100) for i in range(5, 95, 5)]}}

POOL_SIZE = len(LAYERS_POOL)

Event = namedtuple('event', ['type', 'stage'])
fake = faker.Faker()


class Layer():
    """
    Single layer class with compatibility checking
    """

    def __init__(self, layer_type, previous_layer=None, next_layer=None, classes=None):
        self._classes = classes
        self.config = {}
        self.type = layer_type

        self._init_parameters_()
        self._check_compatibility_(previous_layer, next_layer)

    def _init_parameters_(self):
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

    def _check_compatibility_(self, previous_layer, next_layer):
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


class Individ():
    """
    Invidiv class for text data types
    """
    # TODO: add support for different data types
    # TODO: add support for different task types
    # TODO: all data types as subclass?
    # TODO: network parser to avoid layer compatibility errors and shape errors

    def __init__(self, stage, data_type='text', task_type='classification', parents=None, freeze=None, **kwargs):
        """
        Create individ randomly or with its parents
        parents: set of two individ objects
        kwargs: dictionary with manualy specified parameters like number of classes,
        training parameters, etc
        """
        self._stage = stage
        self._data_processing_type = data_type
        self._task_type = task_type
        # TODO: freeze training or data parameters of individ and set manualy
        self._freeze = freeze
        self._parents = parents
        self.options = kwargs
        self._history = [Event('Init', stage)]
        self._name = fake.name().replace(' ', '_') + '_' + str(stage)
        self._architecture = []
        self._data_processing = None
        self._training_parameters = None
        self.shape_structure = None
        self._layers_number = 0
        self._result = 0.0

        if self._parents is None:
            self._random_init_()
        else:
            self._init_with_crossing_()

        self._check_compatibility()

    def _init_layer_(self, layer):
        """
        Return layer according its configs as keras object
        """
        if layer.type == 'lstm':
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

    def _random_init_(self):
        """
        At first, we set probabilities pool and the we change
        this uniform distribution according to previous layer
        """
        if self._architecture:
            self._architecture = []

        # choose number of layers
        self._layers_number = np.random.randint(1, 10)

        # initialize pool of uniform probabilities
        probabilities_pool = np.full((POOL_SIZE), 1 / POOL_SIZE)

        # dict of layers and their indexes
        # indexes are used to change probabilities pool
        pool_index = {i: name for i, name in enumerate(LAYERS_POOL.keys())}

        # layers around current one
        previous_layer = None
        next_layer = None
        tmp_architecture = []

        # Create structure
        for i in range(self._layers_number):
            # choose layer index
            tmp_layer = np.random.choice(list(pool_index.keys()), p=probabilities_pool)
            tmp_architecture.append(tmp_layer)

            # increase probability of this layer type
            probabilities_pool[tmp_layer] *= 2
            probabilities_pool /= probabilities_pool.sum()

        # generate architecture
        for i, name in enumerate(tmp_architecture):
            if i != 0:
                previous_layer = pool_index[tmp_architecture[i - 1]]
            if i < len(tmp_architecture) - 1:
                next_layer = pool_index[tmp_architecture[i + 1]]
            if i == len(tmp_architecture) - 1:
                next_layer = 'last_dense'

            layer = Layer(pool_index[name], previous_layer, next_layer)
            self._architecture.append(layer)

        if self._data_processing_type == 'text':
            # Push embedding for texts
            layer = Layer('embedding')
            self._architecture.insert(0, layer)
        else:
            raise Exception('Unsupported data type')

        if self._task_type == 'classification':
            # Add last layer
            layer = Layer('last_dense', classes=self.options['classes'])
            self._architecture.append(layer)
        else:
            raise Exception('Unsupported task type')

        self._training_parameters = self._random_init_training_()
        self._data_processing = self._random_init_data_processing()

    def _init_with_crossing_(self):
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

        self._history.append(Event('Birth', self._stage))

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
            changes_layer = np.random.choice([i for i in range(1, len(self._architecture) - 1)])
            alter_layer = np.random.choice([i for i in range(1, len(mother.architecture) - 1)])

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

    def _random_init_training_(self):
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
        if not self._architecture:
            raise Exception('Not initialized yet')

        if self._data_processing_type == 'text':
            data_tmp = {}
            data_tmp['vocabular'] = self._architecture[0].config['vocabular']
            data_tmp['sentences_length'] = self._architecture[0].config['sentences_length']
            data_tmp['classes'] = self.options['classes']

        return data_tmp

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
                network_graph.add(self._init_layer_(layer))
            except Exception as e:
                # in some cases shape of previous output can be less than kernel size of cnn
                # add same padding to avoid this problem
                if str(e)[:45] == 'Negative dimension size caused by subtracting':
                    layer.config['padding'] = 'same'
                    network_graph.add(self._init_layer_(layer))
                else:
                    raise

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

    def mutation(self, stage):
        """
        Darwin was right. Change some part of individ with some probability
        """
        mutation_type = np.random.choice(['architecture', 'train', 'all'], p=(0.45, 0.45, 0.1))
        self._history.append(Event('Mutation', stage))

        if mutation_type == 'architecture':
            # all - change the whole net
            mutation_size = np.random.choice(['all', 'part', 'parameters'], p=(0.3, 0.3, 0.4))

            if mutation_size == 'all':
                # If some parts of the architecture change, the processing of data must also change
                self._random_init_()

            elif mutation_size == 'part':
                # select layer except the first and the last one - embedding and dense(*)
                mutation_layer = np.random.choice([i for i in range(1, len(self._architecture) - 1)])

                # find next layer to avoid incopabilities in neural architecture
                next_layer = self._architecture[mutation_layer + 1]
                new_layer = np.random.choice(list(LAYERS_POOL.keys()))
                layer = Layer(new_layer, next_layer=next_layer)

                self._architecture[mutation_layer] = layer

            elif mutation_size == 'parameters':
                # select layer except the first and the last one - embedding and dense(3)
                mutation_layer = np.random.choice([i for i in range(1, len(self._architecture) - 1)])

                # find next layer to avoid incopabilities in neural architecture
                next_layer = self._architecture[mutation_layer + 1]
                new_layer = self._architecture[mutation_layer].type

                self._architecture[mutation_layer] = Layer(new_layer, next_layer=next_layer)

        elif mutation_type == 'train':
            mutation_size = np.random.choice(['all', 'part'], p=(0.3, 0.7))

            if mutation_size == 'all':
                self._training_parameters = self._random_init_training_()

            elif mutation_size == 'part':
                mutation_parameter = np.random.choice(list(TRAINING))
                new_training = self._random_init_training_()
                self._training_parameters[mutation_parameter] = new_training[mutation_parameter]

        elif mutation_type == 'all':
            # change the whole individ - similar to death and rebirth
            self._random_init_()

    def crossing(self, other, stage):
        """
        Create new object as crossing between this one and the other
        """
        new_individ = Individ(stage=stage, classes=2, parents=(self, other))
        return new_individ

    def get_layer_number(self):
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
