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

from ..constants import GENERAL, LAYERS_POOL, SPECIAL, TRAINING


def parse_mutation_const():
    tmp_probability = 1
    mutations_probability = {mutation: tmp_probability for mutation in GENERAL['mutation_type']}

    return mutations_probability


def parse_layer_const():
    """
    Parse all available layers and set initial probability
    """
    # uniform distribution
    tmp_probability = 1  # / len(LAYERS_POOL)

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
    tmp_probability = 1  # / len(GENERAL['layers_number'])
    layers_number_probability = {value: tmp_probability for value in GENERAL['layers_number']}

    return layers_number_probability


def parse_training_const():
    """
    Parse all available training parameters and set initial probability
    """
    training_parameters_probability = {}

    for parameter in TRAINING:
        tmp_probability = 1  # / len(TRAINING[parameter])
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
    return 0.423 ** (1 / (1 + abs(index_of_selected_value - x)))


class Distribution():
    """
    Here we evolve our own distribution for all population. At the end of the evolution
    it is possible to generate individs from this distribution
    """
    _mutations_probability = parse_mutation_const()
    _layers_probability = parse_layer_const()
    _layers_parameters_probability = parse_layer_parameter_const()
    _layers_number_probability = parse_layers_number()
    _training_parameters_probability = parse_training_const()
    # True value of this parameter leads to fast convergence
    # TODO: options
    _appeareance_increases_probability = False
    _diactivated_layers = []

    def _increase_layer_probability(self, layer):
        self._layers_probability[layer] += 0.1

    def _increase_layer_parameters_probability(self, layer, parameter, value):
        a = list(self._layers_parameters_probability[layer][parameter])

        a.sort()
        index_of_selected_value = a.index(value)

        for i, value in enumerate(a):
            self._layers_parameters_probability[layer][parameter][value] += kernel(i, index_of_selected_value)

    def _increase_training_parameters(self, parameter, value):
        a = list(self._training_parameters_probability[parameter])

        a.sort()
        index_of_selected_value = a.index(value)

        for i, value in enumerate(a):
            self._training_parameters_probability[parameter][value] += kernel(i, index_of_selected_value)

    @classmethod
    def mutation(cls):
        """
        Get random mutation type
        """
        tmp = {key: value for key, value in cls._mutations_probability.items()}
        a = list(tmp)

        # we should normalize list of probabilities
        p = np.array(list(tmp.values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        return choice

    @classmethod
    def layer(cls):
        """
        Get the random layer's type
        """
        # We use dictionary of probabilities and exclude disactivated layers
        tmp = {key: value for key, value in cls._layers_probability.items() if key not in cls._diactivated_layers}
        a = list(tmp)

        # we should normalize list of probabilities
        p = np.array(list(tmp.values()))

        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if cls._appeareance_increases_probability:
            # now we increase the probability of this layer to be appear
            cls._increase_layer_probability(cls, choice)

        return choice

    @classmethod
    def layer_parameters(cls, layer, parameter):
        """
        Get random parameters for the layer
        """
        a = list(cls._layers_parameters_probability[layer][parameter])
        if not a:
            return None

        # we should normalize list of probabilities
        p = np.array(list(cls._layers_parameters_probability[layer][parameter].values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if cls._appeareance_increases_probability:
            # now one important thing - imagine parameters as a field of values
            # we chose one value, and now we want to increase the probability of this value
            # but we also should increase probabilities of near values
            cls._increase_layer_parameters_probability(cls, layer, parameter, choice)

        return choice

    @classmethod
    def layers_number(cls):
        """
        Get the number of layers
        """
        a = list(cls._layers_number_probability)

        # we should normalize list of probabilities
        p = np.array(list(cls._layers_number_probability.values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if cls._appeareance_increases_probability:
            # now one important thing - imagine parameters as a field of values
            # we chose one value, and now we want to increase the probability of this value
            # but we also should increase probabilities of near values
            a.sort()
            index_of_selected_value = a.index(choice)

            for i, value in enumerate(a):
                cls._layers_number_probability[value] += kernel(i, index_of_selected_value)

        return choice

    @classmethod
    def training_parameters(cls, parameter):
        """
        Get the training parameter
        """
        a = list(cls._training_parameters_probability[parameter])

        # we should normalize list of probabilities
        p = np.array(list(cls._training_parameters_probability[parameter].values()))
        p = p / p.sum()

        choice = np.random.choice(a, p=p)

        if cls._appeareance_increases_probability:
            # now one important thing - imagine parameters as a field of values
            # we chose one value, and now we want to increase the probability of this value
            # but we also should increase probabilities of near values
            cls._increase_training_parameters(cls, parameter, choice)

        return choice

    @classmethod
    def parse_architecture(cls, individ):
        """
        Parse architecture and increase the probability of its elements
        """
        for block in individ.architecture:
            if block.type != 'input' and \
               block.type != 'embedding' and \
               block.type != 'last_dense' and \
               block.type != 'flatten':
                cls._increase_layer_probability(cls, block.type)

                for layer in block.layers:
                    for parameter in layer.config:
                        cls._increase_layer_parameters_probability(cls, block.type, parameter, layer.config[parameter])

        for parameter in individ.training_parameters:
            cls._increase_training_parameters(cls, parameter, individ.training_parameters[parameter])

    @classmethod
    def get_probability(cls):
        """
        Get dictionary of probabilities
        """
        return cls._layers_parameters_probability, cls._layers_probability

    @classmethod
    def set_layer_status(cls, layer, active=True):
        """
        Activate or disactivate one type of layer in case of incompatibilities this data type or task
        """
        if layer in cls._diactivated_layers and active is True:
            cls._diactivated_layers.remove(layer)

        elif layer not in cls._diactivated_layers and active is False:
            cls._diactivated_layers.append(layer)
