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
import copy
import numpy as np

from ..constants import FAKE, GENERAL, LAYERS_POOL, SPECIAL, TRAINING


def parse_mutation_const():
    tmp_probability = 1
    mutations_probability = {mutation: tmp_probability for mutation in GENERAL['mutation_type']}
    # TODO: hm, some base?
    mutations_probability['remove_connection'] = 0.2
    mutations_probability['remove_layer'] = 0.15

    return mutations_probability


def parse_layer_const(probability_to_modify=None):
    """
    Parse all available layers and set initial probability
    """
    if probability_to_modify is not None:
        layers_probability = dict(probability_to_modify)

        # for special_layers in CUSTOM_LAYERS_MAP:
        #     layers_probability[special_layers] = 1

    else:
        # uniform
        tmp_probability = 1

        # probability of each layer type
        layers_probability = {layer: tmp_probability for layer in LAYERS_POOL}

    return layers_probability


def parse_layer_parameter_const():
    """
    Parse all available layer's parameters and set initial probability
    """
    layers_parameters_probability = {}

    for layer in LAYERS_POOL:
        layers_parameters_probability[layer] = {}

        # for each layer's parameter we set probability of its value
        for parameter in LAYERS_POOL[layer]:
            tmp_probability = 1  # / len(LAYERS_POOL[layer][parameter])
            layers_parameters_probability[layer][parameter] = {value: tmp_probability
                                                               for value in LAYERS_POOL[layer][parameter]}

    for layer in SPECIAL:
        layers_parameters_probability[layer] = {}

        # for each layer's parameter we set probability of its value
        for parameter in SPECIAL[layer]:
            tmp_probability = 1  # / len(SPECIAL[layer][parameter])
            layers_parameters_probability[layer][parameter] = {value: tmp_probability
                                                               for value in SPECIAL[layer][parameter]}

    return layers_parameters_probability


def parse_layers_number():
    """
    Parse all available number of layers and set initial probability
    """
    tmp_probability = 1
    layers_number_probability = {value: tmp_probability for value in GENERAL['layers_number']}

    return layers_number_probability


def parse_training_const():
    """
    Parse all available training parameters and set initial probability
    """
    training_parameters_probability = {}

    for parameter in TRAINING:
        tmp_probability = 1
        training_parameters_probability[parameter] = {value: tmp_probability for value in TRAINING[parameter]}

    return training_parameters_probability


def kernel(x, index_of_selected_value, coef=0.423):
    """
    Allows to change the distribution of the element according to the
    distance to selected element

    Arguments:
        x {int} -- index of element in the distribution
        index_of_selected_value {int} -- index of element, which is selected in the distribution

    Returns:
        float -- coefficient for the x element probability
    """
    return 1 - coef ** (1 / (1 + abs(index_of_selected_value - x)))


class Distribution():
    """
    Here we evolve our own distribution for all population. At the end of the evolution
    it is possible to generate individs from this distribution
    """
    def __init__(self):
        self._mutations_probability = parse_mutation_const()
        self._layers_probability = parse_layer_const()
        self._layers_parameters_probability = parse_layer_parameter_const()
        self._layers_number_probability = parse_layers_number()
        self._training_parameters_probability = parse_training_const()

        # True value of this parameter leads to fast convergence
        # TODO: options
        self._appeareance_increases_probability = False
        self._diactivated_layers = []
        self._GENERAL = dict(GENERAL)
        self._LAYERS_POOL = dict(LAYERS_POOL)
        self.CUSTOM_LAYERS_MAP = dict()
        self._SPECIAL = dict(SPECIAL)
        self._TRAINING = dict(TRAINING)

    def reset(self):
        self._mutations_probability = parse_mutation_const()
        self._layers_probability = parse_layer_const()
        self._layers_parameters_probability = parse_layer_parameter_const()
        self._layers_number_probability = parse_layers_number()
        self._training_parameters_probability = parse_training_const()
        self._appeareance_increases_probability = False
        self._diactivated_layers = []
        self.CUSTOM_LAYERS_MAP = {}

    def _increase_layer_probability(self, layer):
        #self._layers_probability[layer] += 0.1

        a = list(self._layers_probability)

        # stupid hack to solve None ordering
        if isinstance(a[0], str):
            gag = '0'
        else:
            gag = 0

        a = sorted(a, key=lambda x: x if x else gag)
        index_of_selected_value = a.index(layer)

        for i, layer in enumerate(a):
            self._layers_probability[layer] += kernel(i, index_of_selected_value, 0.95)

    def _update_layer_probability_pool(self):
        self._layers_probability = parse_layer_const(self._layers_probability)

    def _increase_layer_parameters_probability(self, layer, parameter, value):
        a = list(self._layers_parameters_probability[layer][parameter])

        # stupid hack to solve None ordering
        if isinstance(a[0], str):
            gag = '0'
        else:
            gag = 0

        a = sorted(a, key=lambda x: x if x else gag)
        index_of_selected_value = a.index(value)

        for i, value in enumerate(a):
            self._layers_parameters_probability[layer][parameter][value] += kernel(i, index_of_selected_value)

    def _increase_training_parameters(self, parameter, value):
        a = list(self._training_parameters_probability[parameter])

        # stupid hack to solve None ordering
        if isinstance(a[0], str):
            gag = '0'
        else:
            gag = 0

        a = sorted(a, key=lambda x: x if x else gag)
        index_of_selected_value = a.index(value)

        for i, value in enumerate(a):
            self._training_parameters_probability[parameter][value] += kernel(i, index_of_selected_value)

    def mutation(self):
        """
        Get random mutation type
        """
        tmp = {key: value for key, value in self._mutations_probability.items()}
        a = list(tmp)

        # we should normalize list of probabilities
        p = np.array(list(tmp.values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        return choice

    def layer(self):
        """
        Get the random layer's type
        """
        # We use dictionary of probabilities and exclude disactivated layers
        tmp = {key: value for key, value in self._layers_probability.items() if key not in self._diactivated_layers}
        a = list(tmp)

        # we should normalize list of probabilities
        p = np.array(list(tmp.values()))

        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if self._appeareance_increases_probability:
            # now we increase the probability of this layer to be appear
            self._increase_layer_probability(choice)

        return choice

    def layer_parameters(self, layer, parameter):
        """
        Get random parameters for the layer
        """
        a = list(self._layers_parameters_probability[layer][parameter])
        if not a:
            return None

        # we should normalize list of probabilities
        p = np.array(list(self._layers_parameters_probability[layer][parameter].values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if self._appeareance_increases_probability:
            # now one important thing - imagine parameters as a field of values
            # we chose one value, and now we want to increase the probability of this value
            # but we also should increase probabilities of near values
            self._increase_layer_parameters_probability(layer, parameter, choice)

        return choice

    def layers_number(self):
        """
        Get the number of layers
        """
        a = list(self._layers_number_probability)

        # we should normalize list of probabilities
        p = np.array(list(self._layers_number_probability.values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if self._appeareance_increases_probability:
            # now one important thing - imagine parameters as a field of values
            # we chose one value, and now we want to increase the probability of this value
            # but we also should increase probabilities of near values
            # stupid hack to solve None ordering
            if isinstance(a[0], str):
                gag = '0'
            else:
                gag = 0

            a = sorted(a, key=lambda x: x if x else gag)
            index_of_selected_value = a.index(choice)

            for i, value in enumerate(a):
                self._layers_number_probability[value] += kernel(i, index_of_selected_value)

        return choice

    def training_parameters(self, parameter):
        """
        Get the training parameter
        """
        a = list(self._training_parameters_probability[parameter])

        # we should normalize list of probabilities
        p = np.array(list(self._training_parameters_probability[parameter].values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if self._appeareance_increases_probability:
            # now one important thing - imagine parameters as a field of values
            # we chose one value, and now we want to increase the probability of this value
            # but we also should increase probabilities of near values
            self._increase_training_parameters(parameter, choice)

        return choice

    def parse_architecture(self, individ):
        """
        Parse architecture and increase the probability of its elements
        """
        for block in individ.architecture:
            if block.type != 'input' and \
               block.type != 'embedding' and \
               block.type != 'last_dense' and \
               block.type != 'flatten':
                self._increase_layer_probability(block.type)

                for layer in block.layers:
                    for parameter in layer.config:
                        self._increase_layer_parameters_probability(block.type, parameter, layer.config[parameter])

        for parameter in individ.training_parameters:
            self._increase_training_parameters(parameter, individ.training_parameters[parameter])

    def get_probability(self):
        """
        Get dictionary of probabilities
        """
        return self._layers_parameters_probability, self._layers_probability

    def set_layer_status(self, layer, active=True):
        """
        Activate or disactivate one type of layer in case of incompatibilities this data type or task
        """
        if layer in self._diactivated_layers and active is True:
            self._diactivated_layers.remove(layer)

        elif layer not in self._diactivated_layers and active is False:
            self._diactivated_layers.append(layer)

    def register_new_layer(self, new_layer):
        new_name = 'CUSTOM_{}_{}_{}'.format(FAKE.name().replace(' ', '_'), new_layer.size, new_layer.width)
        self._LAYERS_POOL[new_name] = {}
        self.CUSTOM_LAYERS_MAP[new_name] = copy.deepcopy(new_layer)
