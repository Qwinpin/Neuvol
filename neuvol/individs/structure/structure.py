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
        """
        Class of individ architecture
        Growing steps is applied to the origin matrix
        Mutations store as a additional pool, which we be applied for initializatio of the graph
        This allows to calculate the contribution of each mutation and also remove bad mutations

        There are two version of some methods, private one exists to work with mutations without
        explicit changes of origin matrix. Public methods work with matrix and other properties explicitly
        and are used for growing changes.
        """
        self._matrix = None  # matrix of layers connections
        # matrix and layers indexes should be updated after each mutation and changes of the network
        # to avoid excess computations updated version stored in _matrix_updated
        self._matrix_mutated = None
        self._matrix_updated = False

        self.branchs_end = {}  # last layers of the each branch (indexes)
        self.branchs_counter = [1]

        # self.layers_index = {}  # layer index in matrix and its instance
        self._layers_index_reverse = {}  # the same as previous, reversed key-value
        self._layers_index_reverse_mutated = None
        self._layers_index_reverse_updated = False

        self.mutations_pool = []  # list of all mutations, which will be applied for initialization

    def _register_new_layer(self, matrix, layers_index_reverse, new_layer):
        """Add new layer to the indexers and increase the size of layers matrix
        NOTE: New connection will be added in outer scope

        Args:
            new_layer_name {str} - name of new layer

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            int - index of new added layer
        """
        # create copy of the matrix with shape + 1
        _matrix = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
        _matrix[:matrix.shape[0], :matrix.shape[1]] = matrix

        _layers_index_reverse = layers_index_reverse

        # self.layers_indexes[new_layer] = len(self.layers_indexes)
        _layers_index_reverse[len(_layers_index_reverse)] = new_layer

        return _matrix, _layers_index_reverse, len(self._layers_index_reverse) - 1

    def _add_layer(self, matrix, layers_index_reverse, branchs_end, layer, branch, branch_out=None):
        """
        Add layer to the last layer of the branch

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            layer {instance of the Layer} - layer to add
            branch {int} - number of the branch to be connected to
            branch_out {int} - number of the branch after this new layer: if branch is splitted

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
        """
        # index of the layer to add to
        add_to = branchs_end[branch]

        matrix, layers_index_reverse, index = self._register_new_layer(matrix, layers_index_reverse, layer)
        matrix[add_to, index] = 1

        # change output branch if branch slitted
        if branch_out is None:
            branch_out = branch

        branchs_end[branch_out] = index

        return matrix, layers_index_reverse, branchs_end

    def _inject_layer(self, matrix, layers_index_reverse, branchs_end, branchs_counter, layer, before_layer_index, after_layer_index):
        """
        Add new layer between two given layers or to a one layer

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            branchs_counter {list{int}} - array of all branchs currently used
            layer {instance of the Layer} - layer to add
            before_layer_index {int} - index of the layer, to which new layer will be connected
            after_layer_index {int} - index of the layer, which will be connected to new layer

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        matrix, layers_index_reverse, index = self._register_new_layer(matrix, layers_index_reverse, layer)

        if after_layer_index is None:
            # generate new branch index, which was not used
            branch_to_create = [i for i in range(1, (1 + len(branchs_counter) + 1))
                                if i not in branchs_counter][0]
            branchs_counter.append(branch_to_create)
            branchs_end[branch_to_create] = index
        else:
            matrix[index, after_layer_index] = 1

        matrix[before_layer_index, index] = 1

        return matrix, layers_index_reverse, branchs_end, branchs_counter

    def _add_connection(self, matrix, before_layer_index, after_layer_index):
        """
        Add connection between two layer. Does not add new layer

        Args:
            matrix {np.array{int}} - matrix of layers connections
            before_layer_index {int} - index of the layer, from which connection started
            after_layer_index {int} - index of the layer, to which connection will be added

        Return:
            np.array(N, N) - new matrix of connections
        """
        matrix[before_layer_index, after_layer_index] = 1

        return matrix

    def _merge_branchs(self, matrix, layers_index_reverse, branchs_end, branchs_counter, layer, branchs):
        """
        Concat a set of branchs to one single layer

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            branchs_counter {list{int}} - array of all branchs currently used
            layer {instance of the Layer} - layer, which will be added after concatenation
            branchs {list{int}} -- list of branchs to concatenate

        Returns:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        adds_to = [branchs_end[branch] for branch in branchs]
        matrix, layers_index_reverse, index = self._register_new_layer(matrix, layers_index_reverse, layer)

        for branch in adds_to:
            matrix[branch, index] = 1

        for branch in branchs:
            del branchs_end[branch]

            # remove branches, which will be concatenated
            tmp_index = branchs_counter.index(branch)
            branchs_counter.pop(tmp_index)

        branch_new = [i for i in range(1, (1 + len(branchs_counter) + 1)) if i not in branchs_counter][0]
        branchs_counter.append(branch_new)
        branchs_end[branch_new] = index

        return matrix, layers_index_reverse, branchs_end, branchs_counter

    def _split_branch(self, matrix, layers_index_reverse, branchs_end, branchs_counter, layers, branch):
        """
        Split branch into two new branchs

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            branchs_counter {list{int}} - array of all branchs currently used
            layers {list{instance of the Layer}} - layers, which form new branchs
            branch {int} - branch, which should be splitted

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        add_to = branchs_end[branch]

        indexes = []
        for layer in layers:
            matrix, layers_index_reverse, index = self._register_new_layer(matrix, layers_index_reverse, layer)
            indexes.append(index)

        list_of_branchs_to_create = [i for i in range(1, (len(indexes) + len(branchs_counter) + 1))
                                     if i not in branchs_counter]
        branchs_counter.extend(list_of_branchs_to_create)

        for i, layer_index in enumerate(indexes):
            matrix[add_to, layer_index] = 1

            branchs_end[list_of_branchs_to_create[i]] = layer_index

        # remove splitted branch
        del branchs_end[branch]
        branchs_counter.pop(branchs_counter.index(branch))

        return matrix, layers_index_reverse, branchs_end, branchs_counter

    def _add_mutation(self, mutation):
        """
        Add mutation to the pool of all mutations, which will be applied one by one on calling matrix or layer indexes

        Args:
            mutation {Mutation instance} - mutation to add
        """
        self.mutations_pool.append(mutation)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False

    def add_layer(self, layer, branch, branch_out=None):
        """
        Public method of _add_layer, which add layer explicitly and permanently

        Args:
            layer {instance of the Layer} - layer to add
            branch {int} - number of the branch to be connected to
            branch_out {int} - number of the branch after this new layer: if branch is splitted
        """
        self._matrix, self._layers_index_reverse, self.branchs_end = self._add_layer(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, layer,
            branch, branch_out)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False

    def inject_layer(self, layer, before_layer_index, after_layer_index):
        """
        Public method of _inject_layer, which inject layer explicitly and permanently

        Args:
            layer {instance of the Layer} - layer to add
            before_layer_index {int} - index of the layer, to which new layer will be connected
            after_layer_index {int} - index of the layer, which will be connected to new layer
        """
        self._matrix, self._layers_index_reverse, self.branchs_end, self.branchs_counter = self._inject_layer(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, self.branchs_counter, layer,
            before_layer_index, after_layer_index)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False

    def add_connection(self, before_layer_index, after_layer_index):
        """
        Public method of _add_connection, which add connection explicitly and permanently

        Args:
            before_layer_index {int} - index of the layer, from which connection started
            after_layer_index {int} - index of the layer, to which connection will be added
        """
        self._matrix = self._add_connection(self._matrix, before_layer_index, after_layer_index)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False

    def merge_branchs(self, layer, branchs=None):
        """
        Public method of _merge_branchs, which merge branchs explicitly and permanently

        Args:
            layer {instance of the Layer} - layer, which will be added after concatenation
            branchs {list{int}} -- list of branchs to concatenate
        """
        self._matrix, self._layers_index_reverse, self.branchs_end, self.branchs_counter = self._merge_branchs(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, self.branchs_counter, layer, branchs)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False

    def split_branch(self, layers, branch):
        """
        Public method of _split_branch, which split branch explicitly and permanently

        Args:
            layers {list{instance of the Layer}} - layers, which form new branchs
            branch {int} - branch, which should be splitted
        """
        self._matrix, self._layers_index_reverse, self.branchs_end, self.branchs_counter = self._split_branch(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, self.branchs_counter, layers, branch)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False

    def _cyclic_check_dict(self, tree):
        """
        Check if tree in form of dict is cyclic or not

        Args:
            tree {dict} - tree to check

        Return:
            boolean - is cyclic or not
        """
        path = set()

        def visit(vertex):
            path.add(vertex)
            for neighbour in tree.get(vertex, ()):
                if neighbour in path or visit(neighbour):
                    return True
            path.remove(vertex)
            return False

        return any(visit(v) for v in tree)

    def _cyclic_check(self, matrix):
        """
        Check if the architecture is cyclic of not
        Neural network should acyclic

        Args:
            matrix {np.array{int}} - matrix of layer connections

        Return:
            boolean - is cyclic or not
        """
        # generate tree (dict) first
        tree = {}
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if element == 1:
                    if tree.get(i, None) is None:
                        tree[i] = []
                    tree[i].append(j)

        return self._cyclic_check_dict(tree)

    def mutations_applier(self):
        """
        Apply all mutations, which does not create cycle

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        # create copy of properties
        # mutations can lead to a cycle and should be performed with additional checks
        matrix_copy = np.array(self._matrix)
        layers_index_reverse_copy = dict(self._layers_index_reverse)
        branchs_end_copy = dict(self.branchs_end)
        branchs_counter_copy = list(self.branchs_counter)

        for mutation in self.mutations_pool:
            if mutation.mutation_type == 'add_layer':
                layer = mutation.layer
                before_layer_index = mutation.config['before_layer_index']
                after_layer_index = mutation.config['after_layer_index']

                matrix_copy_tmp, layers_index_reverse_copy_tmp, branchs_end_copy_tmp, branchs_counter_copy_tmp = self._inject_layer(
                    matrix_copy, layers_index_reverse_copy,
                    branchs_end_copy, branchs_counter_copy, layer,
                    before_layer_index, after_layer_index)

            elif mutation.mutation_type == 'add_connection':
                before_layer_index = mutation.config['before_layer_index']
                after_layer_index = mutation.config['after_layer_index']

                matrix_copy_tmp = self._add_connection(matrix_copy, before_layer_index, after_layer_index)
                layers_index_reverse_copy_tmp = None
                branchs_end_copy_tmp = None
                branchs_counter_copy_tmp = None

            elif mutation.mutation_type == 'remove_layer':
                pass

            elif mutation.mutation_type == 'remove_connection':
                pass

            # its should be False
            if not self._cyclic_check(matrix_copy_tmp):
                matrix_copy = matrix_copy_tmp
                layers_index_reverse_copy = layers_index_reverse_copy_tmp or layers_index_reverse_copy
                branchs_end_copy = branchs_end_copy_tmp or branchs_end_copy
                branchs_counter_copy = branchs_counter_copy_tmp or branchs_counter_copy
            else:
                matrix_copy_tmp = None
                layers_index_reverse_copy_tmp = None
                branchs_end_copy_tmp = None
                branchs_counter_copy_tmp = None

        return matrix_copy, layers_index_reverse_copy, branchs_end_copy, branchs_counter_copy

    def _update_mutated(self):
        matrix, layers_index_reverse, branchs_end, branchs_counter = self.mutations_applier()

        self._matrix_updated = True
        self._matrix_mutated = matrix

        self._layers_index_reverse_updated = True
        self._layers_index_reverse_mutated = layers_index_reverse

    @property
    def matrix(self):
        # apply all mutations before matrix returning
        if not self._matrix_updated:
            self._update_mutated()

        return self._matrix_mutated

    @property
    def layers_index_reverse(self):
        # apply all mutations before layers indexes returning
        if not self._layers_index_reverse_updated:
            self._update_mutated()

        return self._layers_index_reverse_mutated


class StructureText(Structure):
    def __init__(self, root, embedding):
        """
        Initialize the architecture of the individual

        Args:
            root {instance of the Layer} - input layer type
            embedding {instance of the Layer} - embedding layer type
        """
        super().__init__(root)

        self._matrix = np.zeros((2, 2))

        self._matrix, self._layers_index_reverse, root_index = self._register_new_layer(
            self._matrix,
            self._layers_index_reverse,
            root)

        self._matrix, self._layers_index_reverse, embedding_index = self._register_new_layer(
            self._matrix,
            self._layers_index_reverse,
            embedding)

        self._matrix[root_index, embedding_index] = 1
        self._matrix_pure = self._matrix[:]

        self.branchs_end[1] = embedding_index


class StructureImage(Structure):
    def __init__(self, root):
        super().__init__(root)

        self._matrix = np.zeros((1, 1))
        self._matrix_pure = self._matrix[:]
        self._matrix, self._layers_index_reverse, root_index = self._register_new_layer(
            self._matrix,
            self._layers_index_reverse,
            root)

        self.branchs_end[1] = root_index
