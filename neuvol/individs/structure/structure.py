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


# TODO: rewrite all structure
class Structure:
    def __init__(self, root):
        self.matrix = None  # matrix of layers connections

        self.branchs_end = {}  # last layers of the each branch (indexes)
        self.branches_counter = [1]

        # self.layers_index = {}  # layer index in matrix and its instance
        self.layers_index_reverse = {}  # the same as previous, reversed key-value

        self.layers_count = 0

    def _register_new_layer(self, new_layer):
        """Add new layer to the indexers and increase the size of layers matrix
        NOTE: New connection will be added in outer scope

        Args:
            new_layer_name {str} - name of new layer
        """
        # create copy of the matrix with shape + 1
        tmp_matrix = np.zeros((self.matrix.shape[0] + 1, self.matrix.shape[1] + 1))
        tmp_matrix[:self.matrix.shape[0], :self.matrix.shape[1]] = self.matrix

        self.matrix = tmp_matrix

        # self.layers_indexes[new_layer] = len(self.layers_indexes)
        self.layers_index_reverse[len(self.layers_index_reverse)] = new_layer

        return len(self.layers_index_reverse) - 1

    def add_layer(self, layer, branch, branch_out=None):
        """
        Add layer to the last layer of the branch

        Args:
            layer {instance of the Layer}
            branch {int} - number of the branch to be connected to
            branch_out {int} - number of the branch after this new layer: if branch is splitted
        """
        # index of the layer to add to
        add_to = self.branchs_end[branch]

        index = self._register_new_layer(layer)
        self.matrix[add_to, index] = 1

        # change branch ending
        if branch_out is None:
            branch_out = branch

        self.branchs_end[branch_out] = index

        self.layers_count += 1

    def inject_layer(self, layer, before_layer_index, after_layer_index):
        # TODO: how to resolve acycliс
        index = self._register_new_layer(layer)
        self.matrix[before_layer_index, index] = 1
        self.matrix[index, after_layer_index] = 1

        self.layers_count += 1

    def merge_branches(self, layer, branches=None):
        """
        Concat a set of branches to one single layer

        Args:
            layer {instance of the Layer}

        Keyword Args:
            branches {list{int}} -- list of branches to concat (default: {None})

        Returns:
            str -- return the name of new common ending of the branches
        """
        adds_to = [self.branchs_end[branch] for branch in branches]
        index = self._register_new_layer(layer)

        for branch in adds_to:
            self.matrix[branch, index] = 1

        for branch in branches:
            del self.branchs_end[branch]

            # now remove this branch from list of branch's numbers
            tmp_index = self.branches_counter.index(branch)
            self.branches_counter.pop(tmp_index)

        branch_new = [i for i in range(1, (1 + len(self.branches_counter) + 1)) if i not in self.branches_counter][0]
        self.branches_counter.append(branch_new)
        self.branchs_end[branch_new] = index

    def split_branch(self, layers, branch):
        """
        Split branch into two new branches

        Args:
            layers {list{instance of the Layer}} - layers, which form new branchs
            branch {int} - branch, which should be splitted
        """
        add_to = self.branchs_end[branch]
        indexes = [self._register_new_layer(layer) for layer in layers]

        list_of_branches_to_create = [i for i in range(1, (len(indexes) + len(self.branches_counter) + 1))
                                      if i not in self.branches_counter]
        self.branches_counter.extend(list_of_branches_to_create)

        for i, layer_index in enumerate(indexes):
            self.matrix[add_to, layer_index] = 1

            self.branchs_end[list_of_branches_to_create[i]] = layer_index

        del self.branchs_end[branch]
        self.branches_counter.pop(self.branches_counter.index(branch))

        self.layers_count += len(layers)


class StructureText(Structure):
    def __init__(self, root, embedding):
        """
        Initialize the architecture of the individual

        Args:
            root {instance of the Layer} - input layer type
            embedding {instance of the Layer} - embedding layer type
        """
        super().__init__(root)

        self.matrix = np.zeros((2, 2))

        root_index = self._register_new_layer(root)
        embedding_index = self._register_new_layer(embedding)

        self.matrix[root_index, embedding_index] = 1

        self.branchs_end[1] = embedding_index
        self.layers_count += 2


class StructureImage(Structure):
    def __init__(self, root):
        super().__init__(root)

        self.matrix = np.zeros((1, 1))

        root_index = self._register_new_layer(root)

        self.branchs_end[1] = root_index
        self.layers_count += 1
