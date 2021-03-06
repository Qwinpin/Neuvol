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
import itertools

from .layer import LayerComplex
from ..constants import GENERAL
from ..probabilty_pool.generating_distribution import Distribution


def generate_complex_layers(structure, distribution, number_to_generate=5):
    _structure = copy.deepcopy(structure)
    _structure.matrix = _structure.matrix[:-1, :-1]

    new_chains = structure_parser(_structure, number_to_generate)

    new_graphs = [detect_best_combination(new_chain) for new_chain in new_chains]
    new_graphs = [remove_duplicated_branches(new_chain) for new_chain in new_chains]

    new_graphs_processed = [(build_graph(new_graph, structure.layers_index_reverse), len(new_graph)) for new_graph in new_graphs if new_graph]

    new_layers = [LayerComplex(new_graph[0], new_graph[1], width=width) for new_graph, width in new_graphs_processed]

    for new_layer in new_layers:
        distribution.register_new_layer(new_layer)


def structure_parser(structure, number_to_generate, start_point=None, depth=None):
    depth = depth or GENERAL['graph_parser']['depth']
    # remove first two layer - Input and embedder (in case of text)
    layer_indexes = list(structure.layers_index_reverse.keys())[1:-1]
    layer_indexes_random_sampled = [start_point] if start_point else np.random.choice(layer_indexes, number_to_generate, replace=False if len(layer_indexes) >= number_to_generate else True)

    sublayers_chains = []

    for index in layer_indexes_random_sampled:
        sublayers_chain = sublayer_parser(index, structure.matrix[:-2, :-2], depth, None, 0)

        flatten_sublayers_chain = flatten(sublayers_chain)

        sublayers_chains.append(flatten_sublayers_chain)

    return sublayers_chains


def flatten(chain):
    f = []

    for element in chain:
        if isinstance(element, list):
            f.extend(flatten(element))
        else:
            return [chain]

    return f


def sublayer_parser(start_point, matrix, depth, sub_layer=None, level=0):
    level += 1
    
    if level >= depth:
        return [sub_layer]

    if sub_layer is None:
        sub_layer = []

    sub_layer.append(start_point)
    next_step = np.where(matrix[start_point] == 1)[0]

    if level >= depth:
        return sub_layer

    elif len(next_step) == 1:
        new_chains = sublayer_parser(next_step[0], matrix, depth, list(sub_layer), level)

    elif len(next_step) > 1:
        new_chains = [sublayer_parser(step, matrix, depth, list(sub_layer), level) for step in next_step]

    else:
        return sub_layer

    return new_chains


def detect_best_combination(new_chains, min_size=None):
    min_size = min_size or GENERAL['graph_parser']['min_size']
    
    new_chains = [chain for chain in new_chains if len(chain) >= min_size]
    # if max([len(chain) for chain in new_chains]) < min_size:
    #     return None
    if len(new_chains) == 1:
        return new_chains
    elif len(new_chains) == 0:
        return None

    last_indexes = [i[-1] for i in new_chains]

    frequent_last_index = collections.Counter(last_indexes).most_common(1)[0]
    # if all last indexes are unique - cut the last index
    if frequent_last_index[1] == 1:
        new_chains = [chain[:-1] for chain in new_chains]
        new_subgraph = detect_best_combination(new_chains)

    else:
        new_subgraph = [cut(chain, frequent_last_index[0]) for chain in new_chains if cut(chain, frequent_last_index[0])]

    return new_subgraph

def remove_duplicated_branches(new_chain):
    buffer = []

    for chain in new_chain:
        if chain not in buffer:
            buffer.append(chain)

    return buffer

def cut(chain, node):
    try:
        node_index = chain.index(node)
    except:
        return None

    new_chain = chain[:node_index + 1]
    if len(new_chain) == 1:
        return None

    return new_chain


def build_graph(graph, layers_index_reverse):
    graph_indexes = np.unique(list(itertools.chain(*graph)))
    reindexer = {old_index: new_index for new_index, old_index in enumerate(graph_indexes)}

    selected_graph_layers = {reindexer[layer]: copy.deepcopy(layers_index_reverse[layer]) for layer in reindexer.keys()}

    matrix = np.zeros((len(reindexer), len(reindexer)))
    for path in graph:
        for i, node in enumerate(path[:-1]):
            matrix[reindexer[node], reindexer[path[i + 1]]] = 1

    return matrix, selected_graph_layers
