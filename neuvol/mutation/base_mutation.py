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
import math
import numpy as np

from ..constants import GENERAL
from ..probabilty_pool import Distribution
from ..layer import Layer


def mutator(mutation_type, matrix, layers_types, distribution, config=None, layer=None):
    if mutation_type in MUTATIONS_MAP:
        return MUTATIONS_MAP[mutation_type](
            mutation_type=mutation_type,
            matrix=matrix,
            layers_types=layers_types,
            distribution=distribution,
            config=config,
            layer=layer)
    else:
        raise TypeError()


class MutatorBase:
    """
    Mutator class for textual data
    """

    @staticmethod
    # TODO: check complexity and evaluation time
    def mutate(individ, distribution, mutation_type=None):
        """
        Mutate individ
        """
        # create representation of the individ with all its previous mutations
        matrix = individ.matrix
        layers_names = {index: layer.layer_type for index, layer in individ.layers_index_reverse.items()}

        if mutation_type is None:
            mutation_type = distribution.mutation()

        individ.add_mutation(mutator(mutation_type, matrix, layers_names, distribution))

    @staticmethod
    def grown(individ, distribution):
        # TODO: external probabilities for each dice
        # merging is an absolute genom changing
        merger_dice = _probability_from_branchs(individ, prior_rate=GENERAL['mutation_rate_merge'], delimeter=2)

        # branches, which was merged should not be splitted or grown after
        branchs_exception = []
        if merger_dice:
            # TODO: generate distribution according to results of epochs
            if len(individ.branchs_end.keys()) >= 2:
                # branchs_to_merge_number = np.random.randint(2, len(individ.branchs_end.keys()) + 1)
                number_of_branches = len(individ.branchs_end.keys()) + 1
                branchs_to_merge_number_distribution = np.array(range(2, number_of_branches)) / np.sum(range(2, number_of_branches))
                branchs_to_merge_number = np.random.choice(list(range(2, number_of_branches)), p=branchs_to_merge_number_distribution)
            else:
                branchs_to_merge_number = 0

            branchs_to_merge = np.random.choice(list(individ.branchs_end.keys()), branchs_to_merge_number, replace=False)
            branchs_to_merge = [i for i in branchs_to_merge if i not in branchs_exception]

            layer_type = distribution.layer()
            new_tail = Layer(layer_type, distribution)

            branchs_end_new = individ.merge_branchs(new_tail, branchs_to_merge)

            branchs_exception.append(branchs_end_new)

        # for branch now we need decide split or not
        free_branches = [i for i in individ.branchs_end.keys() if i not in branchs_exception]
        if len(free_branches) == 0:
            return True

        selected_branch = np.random.choice(free_branches, 1)[0]

        split_dice = _probability_from_branchs(individ, prior_rate=GENERAL['mutation_rate_splitting'], delimeter=1.5)

        if split_dice and not merger_dice:
            number_of_splits = np.random.choice(GENERAL['mutation_splitting']['number_of_splits'], p=GENERAL['mutation_splitting']['rates'])
            new_tails = [Layer(distribution.layer(), distribution) for _ in range(number_of_splits)]

            individ.split_branch(new_tails, branch=selected_branch)

        else:
            new_tail = Layer(distribution.layer(), distribution)

            individ.add_layer(new_tail, selected_branch)

        return True


def _probability_from_branchs(individ, prior_rate, delimeter=1):
    number_of_branches = len(individ.branchs_end.keys())

    if number_of_branches > 1:
        probability = (1 - 1 / math.log(number_of_branches, prior_rate)) / delimeter
    else:
        probability = (1 - 1 / math.log(number_of_branches + 1, prior_rate)) / delimeter

    dice = np.random.choice([0, 1], p=[1 - probability, probability])

    return dice


class MutationInjector:
    def __init__(self, mutation_type, matrix, layers_types, distribution, config=None, layer=None, data_load=None):
        if data_load is not None:
            self.load(data_load, distribution)
        else:
            self.config = config or {}
            if mutation_type is None:
                pass
            else:
                self.mutation_type = mutation_type
                self._layer = layer
                self.distribution = distribution
                self._choose_parameters(matrix, layers_types)

    # def _choose_parameters(self, matrix, layers_types, is_add_layer=False):
    #     size = matrix.shape[0]
    #     self.config['before_layer_index'] = self.config.get('before_layer_index', None) or np.random.randint(1, size - 3)

    #     self.config['before_layer_type'] = layers_types[self.config['before_layer_index']]

    #     split_dice = np.random.choice([0, 1], p=[0.9, 0.1]) if is_add_layer else 0
    #     if split_dice:
    #         self.config['after_layer_index'] = None

    #     else:
    #         if self.config.get('after_layer_index', None) is None:
    #             self.config['after_layer_index'] = np.random.randint(self.config['before_layer_index'], size - 3)

    #         if self.config['after_layer_index'] == self.config['before_layer_index']:
    #             self.config['after_layer_index'] = None

    #         else:
    #             self.config['after_layer_type'] = layers_types[self.config['after_layer_index']]

    def _choose_parameters(self, matrix, layers_types, is_add_layer=False):
        size = matrix.shape[0]
        self.config['after_layer_index'] = self.config.get('after_layer_index', None) or np.random.randint(1, size - 3)

        self.config['after_layer_type'] = layers_types[self.config['after_layer_index']]

        split_dice = np.random.choice([0, 1], p=[0.9, 0.1]) if is_add_layer else 0
        if split_dice:
            self.config['before_layer_index'] = None

        else:
            if self.config.get('before_layer_index', None) is None:
                self.config['before_layer_index'] = np.random.randint(self.config['after_layer_index'], size - 3)

            if self.config['before_layer_index'] == self.config['after_layer_index']:
                self.config['before_layer_index'] = None

            else:
                self.config['before_layer_type'] = layers_types[self.config['before_layer_index']]

    @property
    def layer(self):
        return self._layer

    @property
    def after_layer_index(self):
        return self.config['after_layer_index']

    @property
    def before_layer_index(self):
        return self.config['before_layer_index']

    @layer.setter
    def layer(self, layer):
        self._layer = layer

    @after_layer_index.setter
    def after_layer_index(self, index):
        self.config['after_layer_index'] = index

    @before_layer_index.setter
    def before_layer_index(self, index):
        self.config['before_layer_index'] = index

    def dump(self):
        buffer = {}
        buffer['mutation_type'] = self.mutation_type
        if self._layer is None:
            buffer['layer'] = ''
        elif type(self._layer) == int:
            buffer['layer'] = self._layer
        else:
            buffer['layer'] = self._layer.dump()

        buffer['config'] = {}
        buffer['config'] = self.config

        return buffer

    def load(self, data_load, distribution):
        self.mutation_type = data_load['mutation_type']
        if type(data_load['layer']) == str or type(data_load['layer']) == int:
            self._layer = data_load['layer']
        else:
            self._layer = Layer(data_load['layer']['layer_type'], distribution, None, None, None, data_load['layer'])
        self.config = data_load['config']


class MutationInjectorAddLayer(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, distribution, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, distribution, config, layer)

    def _choose_parameters(self, matrix, layers_types):
        super()._choose_parameters(matrix, layers_types)

        self._layer = self._layer or Layer(self.distribution.layer(), self.distribution)


class MutationInjectorAddConnection(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, distribution, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, distribution, config=config, layer=layer)

    def _choose_parameters(self, matrix, layers_types):
        super()._choose_parameters(matrix, layers_types)
        if self.config['after_layer_index'] is None:
            self.config['state'] = 'broken'


class MutationInjectorRemoveLayer(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, distribution, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, distribution, config=config, layer=layer)

    def _choose_parameters(self, matrix, layers_types):
        layer_indexes = list(layers_types.keys())
        layer_to_remove = int(np.random.choice(layer_indexes, size=1)[0])
        self._layer = layer_to_remove


class MutationInjectorRemoveConnection(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, distribution, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, distribution, config=config, layer=layer)


MUTATIONS_MAP = {
    'add_layer': MutationInjectorAddLayer,
    'add_connection': MutationInjectorAddConnection,
    'remove_layer': MutationInjectorRemoveLayer,
    'remove_connection': MutationInjectorRemoveConnection,
}
