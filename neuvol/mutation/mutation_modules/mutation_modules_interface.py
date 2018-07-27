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
from ...layer.layer import LAYERS_POOL


def perform_mutation(individ, mutation_type):
    """
    Perform specific mutation according mutation type
    """
    if mutation_type == 'architecture_part':
        return architecture_part(individ)

    elif mutation_type == 'architecture_parameters':
        return architecture_parameters(individ)

    elif mutation_type == 'training_all':
        return training_all(individ)

    elif mutation_type == 'training_part':
        return training_part(individ)


def architecture_part(individ):
    """
    select layer except the first and the last one - embedding and dense(*)
    """
    mutation_layer = np.random.choice([i for i in range(1, len(individ.architecture) - 1)])

    # find next layer to avoid incopabilities in neural architecture
    next_layer = individ.architecture[mutation_layer + 1]
    new_layer = np.random.choice(list(LAYERS_POOL.keys()))
    block = Block(new_layer, next_block=next_layer)

    individ.architecture[mutation_layer] = block

    return individ


def architecture_parameters(individ):
    """
    Select layer except the first and the last one - embedding and dense(3)
    """
    mutation_layer = np.random.choice([i for i in range(1, len(individ.architecture) - 1)])

    # find next layer to avoid incopabilities in neural architecture
    next_layer = individ.architecture[mutation_layer + 1]
    new_layer = individ.architecture[mutation_layer].type

    individ.architecture[mutation_layer] = Block(new_layer, next_block=next_layer)

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
    new_training = individ.random_init_training()
    individ.training_parameters[mutation_parameter] = new_training[mutation_parameter]

    return individ
