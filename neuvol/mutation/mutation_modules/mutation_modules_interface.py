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

from ...constants import TRAINING
from ...layer.block import Block
from ...probabilty_pool import Distribution


def perform_mutation(individ, mutation_type):
    """
    Perform specific mutation according mutation type
    """
    if individ.data_type == 'text':
        # start slice and end slice
        limitations = [2, 1]

    else:
        limitations = [1, 1]

    if mutation_type == 'architecture_part':
        return architecture_part(individ, limitations)

    elif mutation_type == 'architecture_parameters':
        return architecture_parameters(individ, limitations)

    elif mutation_type == 'training_all':
        return training_all(individ)

    elif mutation_type == 'training_part':
        return training_part(individ)

    elif mutation_type == 'architecture_add':
        return architecture_add_layer(individ, limitations)

    elif mutation_type == 'architecture_remove':
        return architecture_remove_layer(individ, limitations)


def architecture_part(individ, limitations):
    """
    select layer except the first and second one and the last one - input, embedding and dense(*)
    """
    mutation_layer = np.random.choice([i for i in range(limitations[0], len(individ.architecture) - limitations[1])])

    # find next layer to avoid incopabilities in neural architecture
    next_layer = individ.architecture[mutation_layer + 1]
    new_layer = Distribution.layer()
    block = Block(new_layer, next_block=next_layer, **individ.options)

    individ.architecture[mutation_layer] = block

    return individ


def architecture_parameters(individ, limitations):
    """
    select layer except the first and second one and the last one - input, embedding and dense(*)
    """
    mutation_layer = np.random.choice([i for i in range(limitations[0], len(individ.architecture) - limitations[1])])

    # find next layer to avoid incopabilities in neural architecture
    next_layer = individ.architecture[mutation_layer + 1]
    new_layer = individ.architecture[mutation_layer].type

    individ.architecture[mutation_layer] = Block(new_layer, next_block=next_layer, **individ.options)

    return individ


def training_all(individ):
    """
    Mutate all training parameters
    """
    individ.training_parameters = individ.random_init_training()

    return individ


def training_part(individ):
    """
    Choose and mutate only one parameters
    """
    mutation_parameter = np.random.choice(list(TRAINING))
    individ.training_parameters[mutation_parameter] = Distribution.training_parameters(mutation_parameter)

    return individ


def architecture_add_layer(individ, limitations):
    """
    Select random place of new layer and add
    """
    mutation_layer = np.random.choice([i for i in range(limitations[0], len(individ.architecture) - limitations[1])])

    # find next layer to avoid incopabilities in neural architecture
    next_layer = individ.architecture[mutation_layer + 1]
    new_layer = Distribution.layer()
    block = Block(new_layer, next_block=next_layer, **individ.options)

    tmp = individ.architecture
    tmp.insert(mutation_layer, block)

    individ.architecture = tmp

    return individ


def architecture_remove_layer(individ, limitations):
    """
    Remove random layer from architecture
    """
    mutation_layer = np.random.choice([i for i in range(limitations[0], len(individ.architecture) - limitations[1])])

    tmp = individ.architecture
    del tmp[mutation_layer]

    individ.architecture = tmp

    return individ
