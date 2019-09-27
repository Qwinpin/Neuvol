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


def flatten(chain):
    f = []

    for element in chain:
        if isinstance(element, list):
            f.extend(flatten(element))
        else:
            return [chain]

    return f


def structure_parser(structure):
    layer_indexes = list(structure.layers_index_reverse.keys())

    layer_indexes_random_sampled = np.random.choice(layer_indexes, len(layer_indexes) // 2, replace=False)

    sublayers_chains = []

    for index in layer_indexes_random_sampled:
        sublayers_chain = sublayer_parser(index, structure.matrix, None)

        flatten_sublayers_chain = flatten(sublayers_chain)

        sublayers_chains.append(flatten_sublayers_chain)

    return sublayers_chains


def sublayer_parser(start_point, matrix, sub_layer=None):
    if sub_layer is None:
        sub_layer = []

    sub_layer.append(start_point)
    next_step = np.where(matrix[start_point] == 1)[0]

    if len(next_step) == 1:
        new_chains = sublayer_parser(next_step[0], matrix, list(sub_layer))

    elif len(next_step) > 1:
        new_chains = [sublayer_parser(step, matrix, list(sub_layer)) for step in next_step]

    else:
        return sub_layer

    return new_chains