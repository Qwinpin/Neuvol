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


def mutator(mutation_type, matrix, layers_types, config=None, layer=None):
    if mutation_type in MUTATIONS_MAP:
        return MUTATIONS_MAP[mutation_type](
            mutation_type=mutation_type,
            matrix=matrix,
            layers_types=layers_types,
            config=config,
            layer=layer)
    else:
        raise TypeError()


class MutatorBase:
    """
    Mutator class for textual data
    """

    @staticmethod
    def mutate(individ, mutation_type=None):
        """
        Mutate individ
        """
        # create representation of the individ with all its previous mutations
        matrix = individ.matrix
        layers_names = {index: layer.layer_type for index, layer in individ.layers_index_reverse.items()}

        if mutation_type is None:
            mutation_type = Distribution.mutation()

        individ.add_mutation(mutator(mutation_type, matrix, layers_names))

    @staticmethod
    def grown(individ):
        # TODO: external probabilities for each dice
        merger_dice = _probability_from_branchs(individ, prior_rate=GENERAL['mutation_rate_merger'], delimeter=2)

        # branches, which was merged should not be splitted or grown after
        branchs_exception = []
        while merger_dice:
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

            if len(branchs_to_merge) < 2:
                break

            new_tail = Layer(Distribution.layer())

            branchs_end_new = individ.merge_branchs(new_tail, branchs_to_merge)

            merger_dice = _probability_from_branchs(individ, prior_rate=GENERAL['mutation_rate_merger'], delimeter=1)

            branchs_exception.append(branchs_end_new)

        # for each branch now we need decide split or not
        free_branches = [i for i in individ.branchs_end.keys() if i not in branchs_exception]

        split_event = False
        for branch in free_branches:

            split_dice = _probability_from_branchs(individ, prior_rate=GENERAL['mutation_rate_splitting'], delimeter=1.5)

            if split_dice and not split_event:
                number_of_splits = np.random.choice(GENERAL['mutation_splitting']['number_of_splits'], p=GENERAL['mutation_splitting']['rates'])
                new_tails = [Layer(Distribution.layer()) for _ in range(number_of_splits)]

                individ.split_branch(new_tails, branch=branch)
                split_event = True

            else:
                new_tail = Layer(Distribution.layer())

                individ.add_layer(new_tail, branch)

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
    def __init__(self, mutation_type, matrix, layers_types, config=None, layer=None):
        self.mutation_type = mutation_type
        self._layer = layer
        self.config = config or {}

        self._choose_parameters(matrix, layers_types)

    def _choose_parameters(self, matrix, layers_types, is_add_layer=False):
        size = matrix.shape[0]
        self.config['before_layer_index'] = self.config.get('before_layer_index', None) or np.random.randint(1, size - 3)

        self.config['before_layer_type'] = layers_types[self.config['before_layer_index']]

        split_dice = np.random.choice([0, 1], p=[0.9, 0.1]) if is_add_layer else 0
        if split_dice:
            self.config['after_layer_index'] = None

        else:
            if self.config.get('after_layer_index', None) is None:
                self.config['after_layer_index'] = np.random.randint(self.config['before_layer_index'], size - 3)

            if self.config['after_layer_index'] == self.config['before_layer_index']:
                self.config['after_layer_index'] = None

            else:
                self.config['after_layer_type'] = layers_types[self.config['after_layer_index']]

    @property
    def layer(self):
        return self._layer

    @property
    def after_layer_index(self):
        return self.config['after_layer_index']

    @property
    def before_layer_index(self):
        return self.config['before_layer_index']


class MutationInjectorAddLayer(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, config, layer)

    def _choose_parameters(self, matrix, layers_types):
        super()._choose_parameters(matrix, layers_types)

        self._layer = self._layer or Layer(Distribution.layer())


class MutationInjectorAddConnection(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, config=config, layer=layer)

    def _choose_parameters(self, matrix, layers_types):
        super()._choose_parameters(matrix, layers_types)
        if self.config['after_layer_index'] is None:
            self.config['state'] = 'broken'


class MutationInjectorRemoveLayer(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, config=config, layer=layer)

    def _choose_parameters(self, matrix, layers_types):
        layer_indexes = layers_types.keys()

        layer_to_remove = np.random.choice(layer_indexes, size=1)
        self._layer = layer_to_remove


class MutationInjectorRemoveConnection(MutationInjector):
    def __init__(self, mutation_type, matrix, layers_types, config=None, layer=None):
        super().__init__(mutation_type, matrix, layers_types, config=config, layer=layer)


MUTATIONS_MAP = {
    'add_layer': MutationInjectorAddLayer,
    'add_connection': MutationInjectorAddConnection,
    #'remove_layer': MutationInjectorRemoveLayer,
    #'remove_connection': MutationInjectorRemoveConnection,
}
