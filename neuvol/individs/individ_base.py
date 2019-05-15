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
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import adam, RMSprop

from ..constants import EVENT, FAKE, TRAINING
from ..layer.block import Layer
from ..layer.layer import init_layer
from ..probabilty_pool import Distribution
from .structure import Structure
from ..utils import dump


class IndividBase:
    """
    Individ class
    """
    # TODO: add support for different data types
    # TODO: add support for different task types

    def __init__(self, stage, task_type='classification', parents=None, freeze=None, **kwargs):
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
        self.options = kwargs
        self._history = []
        self._name = FAKE.name().replace(' ', '_') + '_' + str(stage)
        self._architecture = []
        self._data_processing = None
        self._training_parameters = None
        self.shape_structure = None
        self._layers_number = 0
        self._result = -1.0

        # skip initialization, if this is tempory individ, that will be used only for load method
        if stage is not None:
            if self._parents is None:
                self._random_init()
                self._history.append(EVENT('Init', stage))
            else:
                # okay, we need some hack to avoid memory leak
                self._parents[0]._parents = None
                self._parents[1]._parents = None

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
        """
        """
        if self._architecture:
            self._architecture = []

        return ...

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

    def layers_imposer(self, net_tail, head, layers_map, arch_map):
        net = net_tail
        source = head

        try:
            target = arch_map[source]
        except KeyError:
            return [net], ''

        # if the next layer is merger - return its net tail
        if target[0][0] == 'm':
            return [net], target[0]

        if len(target) > 1:
            buffer_tails = {branch: layers_map[branch](net) for branch in target}
            buffer = [self.layers_imposer(buffer_tails[branch], branch, layers_map, arch_map) for branch in target]

            buffer_tmp = []
            # unpack
            for sub_branch in buffer:
                sub_net_tails = sub_branch[0]
                for sub_net_tail in sub_net_tails:
                    buffer_tmp.append((sub_net_tail, sub_branch[1]))

            buffer = buffer_tmp
            new_head = buffer[0][1]

            lenghts = int(new_head.split('_')[-1])
            if len(buffer) < lenghts:
                return [branch[0] for branch in buffer], new_head

            # check consistency
            for i in buffer:
                for j in buffer:
                    if i[1] != j[1]:
                        raise ValueError('Inconsistent merger')

            net = concatenate([branch[0] for branch in buffer])

            net, new_head = self.layers_imposer(net, new_head, layers_map, arch_map)

        if 'f' in target[0]:
            net = layers_map[target](net)
            return [net], target[0]

        if len(target) == 1:
            new_head = target[0]
            net = layers_map[target[0]](net[0])

            net, new_head = self.layers_imposer(net, new_head, layers_map, arch_map)

        return [net], new_head

    def init_tf_graph(self):
        """
        Return tensorflow graph, configurated optimizer and loss type of this individ
        """
        if not self._architecture:
            raise Exception('Non initialized net')

        # initialize all layers
        layers_map = {}
        for key, layer in self._architecture.layers.items():
            keras_layer_instance = init_layer(layer)
            layers_map[key] = keras_layer_instance

        starter = 'root'
        network_input = layers_map[starter]

        network_graph = self.layers_imposer(network_input, 'root', layers_map, self._architecture.tree)

        model = Model(inputs=[network_input], outputs=[network_graph])

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

    def save(self):
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
        serial['shape_structure'] = self.shape_structure

        return serial

    def dump(self, path):
        """
        Dump individ as a json object
        """
        dump(self.save(), path)

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
        individ.shape_structure = serial['shape_structure']

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
