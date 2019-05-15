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

from ...constants import EVENT
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
        individ = architecture_part(individ, limitations)

    elif mutation_type == 'architecture_parameters':
        individ = architecture_parameters(individ, limitations)

    elif mutation_type == 'training_all':
        individ = training_all(individ)

    elif mutation_type == 'training_part':
        individ = training_part(individ)

    elif mutation_type == 'architecture_add':
        individ = architecture_add_layer(individ, limitations)

    elif mutation_type == 'architecture_remove':
        individ = architecture_remove_layer(individ, limitations)

    else:
        individ = architecture_part(individ, limitations)

    individ.history = EVENT('Mutation: {}'.format(mutation_type), individ.stage)

    return individ


def architecture_part(individ, limitations):
    """
    select layer except the first and second one and the last one - input, embedding and dense(*)
    """
    mutation_layer = np.random.choice([i for i in range(limitations[0], len(individ.architecture) - limitations[1])])

    # find next layer to avoid incopabilities in neural architecture
    next_layer = individ.architecture[mutation_layer + 1].type
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
    next_layer = individ.architecture[mutation_layer + 1].type
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
    next_layer = individ.architecture[mutation_layer + 1].type
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
    # check if there is no layers to remove
    if len(individ.architecture) <= 4:
        return individ

    try:
        mutation_layer = np.random.choice(
            [i for i in range(limitations[0], len(individ.architecture) - limitations[1])])

        tmp = individ.architecture
        del tmp[mutation_layer]

        individ.architecture = tmp
    except Exception:
        pass

    return individ
