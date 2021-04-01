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
from ..layer import Layer
from ..probabilty_pool import Distribution
from .structure import Structure
from ..utils import dump


class IndividBase:
    """
    Individ class
    """
    def __init__(self, stage, options, finisher, distribution, task_type='classification', parents=None, freeze=None, load_data=None):
        """Create individ randomly or with its parents

        Attributes:
            parents (``IndividBase``): set of two individ objects
            kwargs: dictionary specified parameters like number of classes,
                    training parameters, etc
        """
        self._stage = stage
        self.options = options
        self._finisher = finisher
        self._distribution = distribution
        self._parents = parents
        self._history = []
        self._name = FAKE.name().replace(' ', '_') + '_' + str(stage)
        self._architecture = None

        # fitting metrics
        self._result = None
        self._parameters_number = None

        # generate new architecture or load serialised parameters
        if load_data is not None:
            self.load(load_data)
        else:
            self._random_init()


    def __str__(self):
        return self.name

    def _random_init(self):
        self._architecture = self._random_init_architecture()

    def _random_init_architecture(self):
        """
        Init structure of the individ
        """
        input_layer = Layer('input', self._distribution, options=self.options)
        architecture = Structure(input_layer, self._finisher)

        return architecture

    def cycle_imposer(self, structure):
        """
        Pass through a graph of layers of a neural network with sequential initialization
        """

        # pool of layers, which should be initialised and connected
        layers_pool = [0]

        # pool of initialised layers
        layers_pool_inited = {}

        # pool of broken (invalid) layers) such as inconsistent number of dimensions
        layers_pool_removed = []

        while layers_pool:
            # take first layer in a pool
            layer_index = layers_pool[0]

            # find all connections before this layer
            enter_layers = set(np.where(structure.matrix[:, layer_index] == 1)[0])

            # check if some of previous layers were not initialized
            # that means - we should initialise them first
            not_inited_layers = [i for i in enter_layers if i not in (layers_pool_inited.keys())]
            not_inited_layers_selected = [layer for layer in not_inited_layers if layer not in layers_pool_removed]

            if not_inited_layers_selected:
                # remove layers, which are in pool already
                # this is possible due to complex connections with different orders
                not_inited_layers_selected = [layer for layer in not_inited_layers_selected if layer not in layers_pool]

                # add not initialised layers to the pool
                layers_pool.extend(not_inited_layers_selected)

                # current layer should be shift to the end of the queue
                acc = layers_pool.pop(0)
                layers_pool.append(acc)
                continue

            # take Layer instance of the previous layers
            input_layers = [structure.layers_index_reverse[layer] for layer in enter_layers]

            # layer without rank is broken and we ignore that
            input_layers = [layer for layer in input_layers if layer.config.get('rank', False)]
            enter_layers = [i for i in enter_layers if i not in layers_pool_removed]

            # if curent layer is the Input - initialise without any input connections
            if not input_layers and structure.layers_index_reverse[layer_index].layer_type == 'input':
                inited_layer = structure.layers_index_reverse[layer_index].init_layer(None)

            # detect hanging node - some of mutations could remove connection to the layer
            elif not input_layers:
                layers_pool_removed.append(layers_pool.pop(0))
                continue

            # if there are multiple input connections
            elif len(input_layers) > 1:
                # take initialised instances
                input_layers_inited = [layers_pool_inited[layer] for layer in enter_layers]

                # this case does not require additional processing - all logic is inside Layer instance,
                # which handles multiple connections
                inited_layer = structure.layers_index_reverse[layer_index](input_layers_inited, input_layers)

            else:
                input_layers_inited = [layers_pool_inited[layer] for layer in enter_layers][0]
                inited_layer = structure.layers_index_reverse[layer_index](input_layers_inited, input_layers[0])

            # add new initialised layer
            layers_pool_inited[layer_index] = inited_layer

            # find outgoing connections and add them to the pool
            output_layers = [layer for layer in np.where(structure.matrix[layer_index] == 1)[0]
                                if layer not in layers_pool and layer not in layers_pool_inited.keys()]

            layers_pool.extend(output_layers)

            # remove current layer from the pool
            layers_pool.pop(layers_pool.index(layer_index))

        return layers_pool_inited

    def init_tf_graph_cycle(self):
        """
        Return keras graph
        """
        if not self._architecture:
            raise Exception('Non initialized net')

        # walk over all layers and connect them between each other
        layers_pool = self.cycle_imposer(self._architecture)
        last_layer = max(list(layers_pool.keys()))

        network_head = layers_pool[0]
        network_tail = layers_pool[last_layer]

        model = Model(inputs=[network_head], outputs=[network_tail])

        return model

    def dump(self):
        # serialise the whole individ
        buffer = {}
        structure = self._architecture.dump()
        name = self._name
        stage = self.stage
        options = self.options
        history = self.history

        buffer['structure'] = structure
        buffer['name'] = name
        buffer['stage'] = stage
        buffer['options'] = options
        buffer['history'] = history

        return buffer

    def load(self, data_load):
        # deserialise all parameters
        self._architecture = Structure(None, None, data_load=data_load['structure'], distribution=self._distribution)
        self._name = data_load['name']
        self.stage = data_load['stage']
        self.options = data_load['options']
        self.history =  data_load['history']

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
    def result(self):
        """
        Get the result of the efficiency (f1 or AUC)
        """
        return self._result

    @property
    def result_params(self):
        """
        Get number of parameters in the graph
        """
        return self._parameters_number

    @name.setter
    def name(self, value):
        self._name = value

    @result.setter
    def result(self, value):
        """
        Set new fitness result
        """
        self._result = value

    @result_params.setter
    def result_params(self, value):
        self._parameters_number = value

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
