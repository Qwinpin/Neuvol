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
import collections
import copy
import numpy as np

from ..constants import GENERAL


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


def sublayer_parser(start_point, matrix, sub_layer=None, level=0):
    level += 1

    if level >= GENERAL['graph_parser']['depth']:
        return [sub_layer]

    if sub_layer is None:
        sub_layer = []

    sub_layer.append(start_point)
    next_step = np.where(matrix[start_point] == 1)[0]

    if level >= GENERAL['graph_parser']['depth']:
        return sub_layer

    elif len(next_step) == 1:
        new_chains = sublayer_parser(next_step[0], matrix, list(sub_layer))

    elif len(next_step) > 1:
        new_chains = [sublayer_parser(step, matrix, list(sub_layer)) for step in next_step]

    else:
        return sub_layer

    return new_chains


def cut(chain, node):
    try:
        node_index = chain.index(node)
    except:
        return None

    new_chain = chain[:node_index + 1]
    if len(new_chain) == 1:
        return None

    return new_chain


def detect_best_combination(sublayers):
    if max([len(chain) for chain in sublayers]) < GENERAL['graph_parser']['min_size']:
        return None

    last_indexes = [i[-1] for i in sublayers]

    frequent_last_index = collections.Counter(last_indexes).most_common(1)[0]
    # if all last indexes are unique - cut the last index
    if frequent_last_index[1] == 1:
        sublayers = [chain[:-1] for chain in sublayers]
        new_subgraph = detect_best_combination(sublayers)

    else:
        new_subgraph = [cut(chain, frequent_last_index[0]) for chain in sublayers if cut(chain, frequent_last_index[0])]

    return new_subgraph


def build_graph(graph, layers_index_reverse):
    reindexer = {old_index: new_index for new_index, old_index in enumerate(np.unique(graph))}

    selected_graph_layers = {reindexer[layer]: copy.deepcopy(layers_index_reverse[layer]) for layer in reindexer.keys()}

    matrix = np.zeros((len(reindexer), len(reindexer)))
    for path in graph:
        for i, node in enumerate(path[:-1]):
            matrix[reindexer[node], reindexer[path[i + 1]]] = 1

    return matrix, selected_graph_layers
