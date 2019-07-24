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
from keras.models import Model
from keras.optimizers import adam, RMSprop
import numpy as np

from ..constants import EVENT, FAKE, TRAINING
from ..layer.block import Layer
from ..probabilty_pool import Distribution
from .structure import Structure
from ..utils import dump


class IndividBase:
    """
    Individ class
    """
    # TODO: add support for different data types
    # TODO: add support for different task types

    def __init__(self, stage, options, finisher, task_type='classification', parents=None, freeze=None):
        """Create individ randomly or with its parents

        Attributes:
            parents (``IndividBase``): set of two individ objects
            kwargs: dictionary specified parameters like number of classes,
                    training parameters, etc
        """
        self._stage = stage
        self._data_processing_type = None
        self._task_type = task_type
        # TODO: freeze training or data parameters of individ and set manualy
        self._freeze = freeze
        self._parents = parents
        self.options = options
        self._history = []
        self._name = FAKE.name().replace(' ', '_') + '_' + str(stage)
        self._architecture = None
        self._finisher = finisher
        self._data_processing = None or self.options.get('data_processing', None)
        self._training_parameters = None or self.options.get('training_parameters', None)
        self._layers_number = 0
        self._result = -1.0

        # skip initialization, if this is tempory individ, that will be used only for load method
        if stage is not None:
            if self._parents is None:
                self._random_init()
                self._history.append(EVENT('Init', stage))
            else:
                self._history.append(EVENT('Birth', self._stage))

    def __str__(self):
        return self.name

    def _random_init(self):
        self._architecture = self._random_init_architecture()
        self._data_processing = self._random_init_data_processing()
        self._training_parameters = self._random_init_training()

    def _random_init_architecture(self):
        """
        Init structure of the individ
        """
        input_layer = Layer('input', options=self.options)
        architecture = Structure(input_layer, self._finisher)

        return architecture

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
        return ...

    # def layer_imposer(self):
    #     tails_map = {}
    #     _, columns_number = self.matrix.shape
    #     last_non_zero = 0
    #     for column in range(columns_number):
    #         column_values = self.matrix[:, column]

    #         connections = np.where(column_values == 1)[0]
    #         if not connections.size > 0:
    #             if column == 0:
    #                 tails_map[column] = self.layers_index_reverse[column].init_layer(None)

    #         elif connections.size > 1:
    #             tails_to_call = [tails_map[i] for i in connections]
    #             layers_to_call = [self.layers_index_reverse[i] for i in connections]

    #             tails_map[column] = self.layers_index_reverse[column](tails_to_call, layers_to_call)
    #             last_non_zero = column
    #         else:
    #             tails_map[column] = self.layers_index_reverse[column](tails_map[connections[0]], self.layers_index_reverse[connections[0]])
    #             last_non_zero = column

    #     return tails_map[0], tails_map[last_non_zero]

    def rec_imposer(self, column, tails_map):
        if tails_map.get(column, None) is not None:
            return None

        column_values = self.matrix[:, column]
        connections = np.where(column_values == 1)[0]

        for index in connections:
            if tails_map.get(index, None) is None:
                self.rec_imposer(index, tails_map)

        if not connections.size > 0:
            if column == 0:
                tails_map[column] = self.layers_index_reverse[column].init_layer(None)
            last_non_zero = column

        elif connections.size > 1:
            tails_to_call = [tails_map[i] for i in connections]
            layers_to_call = [self.layers_index_reverse[i] for i in connections]

            tails_map[column] = self.layers_index_reverse[column](tails_to_call, layers_to_call)
            last_non_zero = column

        else:
            tails_map[column] = self.layers_index_reverse[column](tails_map[connections[0]], self.layers_index_reverse[connections[0]])
            last_non_zero = column

        return last_non_zero

    def init_tf_graph(self):
        # TODO: finisher at the of the network
        """
        Return tensorflow graph, configurated optimizer and loss type of this individ
        """
        if not self._architecture:
            raise Exception('Non initialized net')

        tails_map = {}
        last_layer = None

        # walk over all layers and connect them between each other
        for column in range(len(self.layers_index_reverse)):
            last_layer = self.rec_imposer(column, tails_map) or last_layer

        network_head = tails_map[0]
        network_tail = tails_map[last_layer]

        model = Model(inputs=[network_head], outputs=[network_tail])

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

    def _serialize(self):
        """
        Serialize the whole object for further dump
        """
        serial = dict()
        serial['stage'] = self._stage
        serial['data_processing_type'] = self._data_processing_type
        serial['task_type'] = self._task_type
        serial['freeze'] = self._freeze
        serial['parents'] = [parent.save() for parent in self._parents] if self._parents is not None else None
        serial['options'] = self.options
        serial['history'] = self._history
        serial['name'] = self.name
        # TODO: rewrite architectures saver
        # serial['architecture'] = [block.save() for block in self._architecture]
        serial['data_processing'] = self._data_processing
        serial['training_parameters'] = self._training_parameters
        serial['layers_number'] = self._layers_number
        serial['result'] = self._result

        return serial

    def dump(self, path):
        """
        Dump individ as a json object
        """
        dump(self._serialize(), path)

    @classmethod
    def load(cls, serial):
        """
        Base load method. Returns individ
        """
        # replace IndividBase with IndividText or IndividImage according the data type
        individ = cls(None)

        individ._stage = serial['stage']
        individ._data_processing_type = serial['data_processing_type']
        individ._task_type = serial['task_type']
        individ._freeze = serial['freeze']
        if serial['parents'] is not None:
            individ._parents = [IndividBase(None), IndividBase(None)]
            individ._parents = [parent.load(serial['parents'][i]) for i, parent in enumerate(individ._parents)]
        individ.options = serial['options']
        individ._history = [EVENT(*event) for event in serial['history']]
        individ._name = serial['name']

        # TODO: rewrite architectures saver
        # individ._architecture = [Block.load(block) for block in serial['architecture']]

        individ._data_processing = serial['data_processing']
        individ._training_parameters = serial['training_parameters']
        individ._layers_number = serial['layers_number']
        individ._result = serial['result']

        return individ

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
        Get the last stage
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

    @stage.setter
    def stage(self, stage):
        """
        Change stage
        """
        self._stage = stage

    def random_init(self):
        """
        Public method for calling the random initialisation
        """
        self._random_init()

    def random_init_architecture(self):
        """
        Public method for calling the random architecture initialisation
        """
        self._random_init_architecture()

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

    def add_layer(self, layer, branch, branch_out=None):
        """Forward add_layer method of Structure instance

        Args:
            layer {instance of the Layer}
            branch {int} - number of the branch to be connected to
            branch_out {int} - number of the branch after this new layer: if branch is splitted
        """
        self._architecture.add_layer(layer, branch, branch_out=branch_out)

    def merge_branchs(self, layer, branchs=None):
        """
        Forward merge_branches method of Structure instance

        Args:
            layer {instance of the Layer}

        Keyword Args:
            branches {list{int}} -- list of branches to concat (default: {None})

        Returns:
            str -- return the name of new common ending of the branches
        """
        return self._architecture.merge_branchs(layer, branchs=branchs)

    def split_branch(self, layers, branch):
        """
        Forward split_branch method of Structure instance

        Args:
            left_layer {instance of the Layer} - layer, which forms a left branch
            right_layer {instance of the Layer} - layer, which forms a right branch
            branch {int} - branch, which should be splitted
        """
        self._architecture.split_branch(layers, branch)

    def add_mutation(self, mutation):
        """
        Forward add_mutation method of Structure instance

        Args:
            mutation {instance of MutationInjector} - mutation
        """
        self._architecture._add_mutation(mutation)

    def recalculate_shapes(self):
        self._architecture.recalculate_shapes()

    @property
    def matrix(self):
        return self.architecture.matrix

    @property
    def layers_index_reverse(self):
        return self._architecture.layers_index_reverse

    @property
    def layers_counter(self):
        return self._architecture.layers_counter

    @property
    def branches_counter(self):
        return self._architecture.branchs_counter

    @property
    def branchs_end(self):
        return self._architecture.branchs_end
