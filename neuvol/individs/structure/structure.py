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
import copy
import numpy as np

from ...mutation import MutationInjector
from ...layer import Layer
from ...utils import parameters_copy


class Structure:
    #TODO: structure freeze - apply all stable mutations and keep as a new base structure
    def __init__(self, root, finisher, data_load=None, distribution=None):
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
        self._finisher = finisher
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

        if data_load is not None:
            self.load(data_load, distribution)

    @parameters_copy
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

        _layers_index_reverse = dict(layers_index_reverse)

        # self.layers_indexes[new_layer] = len(self.layers_indexes)
        _layers_index_reverse[len(_layers_index_reverse)] = new_layer
        return _matrix, _layers_index_reverse, len(_layers_index_reverse) - 1

    @parameters_copy
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
        print('M', matrix)
        print('B', branchs_end)
        print('B', branch)
        add_to = branchs_end[branch]

        matrix, layers_index_reverse, index = self._register_new_layer(matrix, layers_index_reverse, layer)
        matrix[add_to, index] = 1

        # change output branch if branch slitted
        if branch_out is None:
            branch_out = branch

        branchs_end[branch_out] = index

        return matrix, layers_index_reverse, branchs_end

    @parameters_copy
    def _inject_layer(self, matrix, layers_index_reverse, branchs_end, branchs_counter, layer, before_layer_index, after_layer_index):
        """
        Add new layer between two given layers or to a one layer

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            branchs_counter {list{int}} - array of all branchs currently used
            layer {instance of the Layer} - layer to add
            after_layer_index {int} - index of the layer, to which new layer will be connected
            before_layer_index {int} - index of the layer, which will be connected to new layer

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        matrix, layers_index_reverse, index = self._register_new_layer(matrix, layers_index_reverse, layer)
        if before_layer_index is None:
            print('WW', after_layer_index, before_layer_index, after_layer_index)
            # generate new branch index, which was not used
            branch_to_create = [i for i in range(1, (1 + len(branchs_counter) + 1))
                                if i not in branchs_counter][0]
            branchs_counter.append(branch_to_create)
            branchs_end[branch_to_create] = index
        else:
            print('WW', after_layer_index, before_layer_index, after_layer_index)
            matrix[after_layer_index, before_layer_index] = 0  # remove old direct connection
            matrix[index, before_layer_index] = 1

        matrix[after_layer_index, index] = 1
        return matrix, layers_index_reverse, branchs_end, branchs_counter

    @parameters_copy
    def _remove_layer(self, matrix, layers_index_reverse, branchs_end, branchs_counter, layer_index):
        """
        Remove layer from the structure

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            branchs_counter {list{int}} - array of all branchs currently used
            layer {instance of the Layer} - layer to add

        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        before_layer_indexes = np.where(matrix[:, layer_index] == 1)[0]
        after_layer_indexes = np.where(matrix[layer_index, :] == 1)[0]

        if before_layer_indexes is None:
            # restricted operation - network head could not be removed
            return matrix, layers_index_reverse, branchs_end, branchs_counter

        if after_layer_indexes is not None and len(after_layer_indexes) != 0:
            for i in after_layer_indexes:
                for j in before_layer_indexes:
                    matrix[j, i] = 1
                    matrix[j, layer_index] = 0
                    matrix[layer_index, i] = 0

        else:
            for j in before_layer_indexes:
                matrix[j, layer_index] = 0

        branchs_end_reverse = {value: key for key, value in branchs_end.items()}
        branch_to_remove = branchs_end_reverse.get(layer_index, None)
        if branch_to_remove is not None:
            branchs_counter = [i for i in branchs_counter if i != branch_to_remove]
            del branchs_end[branch_to_remove]

        # del layers_index_reverse[layer_index]

        return matrix, layers_index_reverse, branchs_end, branchs_counter

    @parameters_copy
    def _add_connection(self, matrix, before_layer_index, after_layer_index):
        """
        Add connection between two layer. Does not add new layer

        Args:
            matrix {np.array{int}} - matrix of layers connections
            after_layer_index {int} - index of the layer, from which connection started
            before_layer_index {int} - index of the layer, to which connection will be added

        Return:
            np.array(N, N) - new matrix of connections
        """
        matrix[after_layer_index, before_layer_index] = 1

        return matrix

    @parameters_copy
    def _remove_connection(self, matrix, branchs_end, branchs_counter, before_layer_index, after_layer_index):
        """
        Remove connection between layers in the structure

        Args:
            matrix {np.array{int}} - matrix of layers connections
            after_layer_index {int} - index of the layer, to which new layer will be connected
            before_layer_index {int} - index of the layer, which will be connected to new layer

        Return:
            np.array(N, N) - new matrix of connection
        """
        matrix[after_layer_index, before_layer_index] = 0
        matrix[before_layer_index, after_layer_index] = 0

        # removed connection assumes new branch
        branch_new = [i for i in range(1, (1 + len(branchs_counter) + 1)) if i not in branchs_counter][0]
        branchs_counter.append(branch_new)
        branchs_end[branch_new] = after_layer_index

        return matrix, branchs_end, branchs_counter

    @parameters_copy
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
            try:
                del branchs_end[branch]
            except:
                pass

            # remove branches, which will be concatenated
            try:
                tmp_index = branchs_counter.index(branch)
            except:
                pass

            try:
                branchs_counter.pop(tmp_index)
            except:
                pass

        branch_new = [i for i in range(1, (1 + len(branchs_counter) + 1)) if i not in branchs_counter][0]
        branchs_counter.append(branch_new)
        branchs_end[branch_new] = index

        return matrix, layers_index_reverse, branchs_end, branchs_counter, branch_new

    @parameters_copy
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
            after_layer_index {int} - index of the layer, to which new layer will be connected
            before_layer_index {int} - index of the layer, which will be connected to new layer
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
            after_layer_index {int} - index of the layer, from which connection started
            before_layer_index {int} - index of the layer, to which connection will be added
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
        self._matrix, self._layers_index_reverse, self.branchs_end, self.branchs_counter, branchs_end_new = self._merge_branchs(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, self.branchs_counter, layer, branchs)
        self._matrix_updated = False
        self._layers_index_reverse_updated = False
        return branchs_end_new

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

    def _cyclic_check(self, matrix):
        """
        Check if the architecture is cyclic of not
        Neural network should be acyclic
        Linear combination of connection matrix has to be 0

        Args:
            matrix {np.array{int}} - matrix of layer connections

        Return:
            boolean - is cyclic or not
        """
        for i in range(1, len(matrix) + 5):
            paths = np.linalg.matrix_power(matrix, i)
            if len(np.where(paths.diagonal() != 0)[0]) != 0:
                return True

        return False

    def finisher_applier(self, matrix, layers_index_reverse, branchs_end, branchs_counter):
        """
        Apply all legal mutation and add last layer, defined by finisher

        Args:
            matrix {np.array{int}} - matrix of layers connections
            layers_index_reverse {dict{int, Layer instance}} - indexes of all individ layers
            branchs_end {dict{int, int}} - number of branch and corresponding index of the last layer
            branchs_counter {list{int}} - array of all branchs currently used
        Return:
            np.array(N, N) - new matrix of connection
            dict - new map of layers and their indexes
            dict - new map of branchs and their last layers indexes
            list - new array of branchs indexes
        """
        if matrix is None:
            matrix_copy = np.array(self._matrix)
        else:
            matrix_copy = matrix

        layers_index_reverse_copy = dict(layers_index_reverse) or dict(self._layers_index_reverse)
        branchs_end_copy = dict(branchs_end) or dict(self.branchs_end)
        branchs_counter_copy = list(branchs_counter) or list(self.branchs_counter)

        # current number of branches
        branchs_number = len(branchs_counter)
        print('BBB', branchs_number)

        branchs_to_merge = list(branchs_end_copy.keys())
        if branchs_number > 1:
            matrix_copy_tmp, layers_index_reverse_copy_tmp, branchs_end_copy_tmp, branchs_counter_copy_tmp, _ = self._merge_branchs(
                matrix_copy, layers_index_reverse_copy,
                branchs_end_copy, branchs_counter_copy,
                self._finisher, branchs_to_merge)

        else:
            matrix_copy_tmp, layers_index_reverse_copy_tmp, branchs_end_copy_tmp = self._add_layer(
                matrix_copy, layers_index_reverse_copy,
                branchs_end_copy, self._finisher,
                branchs_to_merge[0])

            branchs_counter_copy_tmp = branchs_counter_copy

        return matrix_copy_tmp, layers_index_reverse_copy_tmp, branchs_end_copy_tmp, branchs_counter_copy_tmp

    def mutations_applier(self, matrix, layers_index_reverse, branchs_end, branchs_counter):
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
        if matrix is None:
            matrix_copy = np.array(self._matrix)
        else:
            matrix_copy = matrix

        layers_index_reverse_copy = dict(layers_index_reverse) or dict(self._layers_index_reverse)
        branchs_end_copy = dict(branchs_end) or dict(self.branchs_end)
        branchs_counter_copy = list(branchs_counter) or list(self.branchs_counter)

        for mutation in self.mutations_pool:
            # increase in complexity due to the imposition of new mutations without remembering past
            if mutation.config.get('state', None) == 'broken':
                continue
            else:
                if mutation.mutation_type == 'add_layer':
                    layer = mutation.layer
                    before_layer_index = mutation.config['before_layer_index']
                    after_layer_index = mutation.config['after_layer_index']

                    matrix_copy_tmp, layers_index_reverse_copy_tmp, branchs_end_copy_tmp, branchs_counter_copy_tmp = self._inject_layer(
                        matrix_copy, layers_index_reverse_copy,
                        branchs_end_copy, branchs_counter_copy, layer,
                        before_layer_index, after_layer_index)

                elif mutation.mutation_type == 'inject_layer':
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
                    layer_index = mutation.layer

                    matrix_copy_tmp, layers_index_reverse_copy_tmp, branchs_end_copy_tmp, branchs_counter_copy_tmp = self._remove_layer(
                        matrix_copy, layers_index_reverse_copy,
                        branchs_end_copy, branchs_counter_copy,
                        layer_index)

                elif mutation.mutation_type == 'remove_connection':
                    pass
                    before_layer_index = mutation.config['before_layer_index']
                    after_layer_index = mutation.config['after_layer_index']
                    matrix_copy_tmp, branchs_end_copy_tmp, branchs_counter_copy_tmp = self._remove_connection(
                        matrix_copy, branchs_end_copy, branchs_counter_copy,
                        before_layer_index, after_layer_index)
                    layers_index_reverse_copy_tmp = None

                # its should be False
                if not self._cyclic_check(matrix_copy_tmp):
                    matrix_copy = matrix_copy_tmp
                    layers_index_reverse_copy = layers_index_reverse_copy_tmp or layers_index_reverse_copy

                    branchs_end_copy = branchs_end_copy_tmp or branchs_end_copy
                    branchs_counter_copy = branchs_counter_copy_tmp or branchs_counter_copy

                    mutation.config['state'] = 'checked'
                else:
                    mutation.config['state'] = 'broken'

                matrix_copy_tmp = None
                layers_index_reverse_copy_tmp = None
                branchs_end_copy_tmp = None
                branchs_counter_copy_tmp = None

        return matrix_copy, layers_index_reverse_copy, branchs_end_copy, branchs_counter_copy

    def _update_mutated(self):
        """
        Update architecture using new mutations
        """
        for mutation in self.mutations_pool:
            if mutation.config.get('state', None) == 'broken':
                mutation.config['state'] = None
        # apply mutations
        matrix, layers_index_reverse, branchs_end, branchs_counter = self.mutations_applier(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, self.branchs_counter)

        # add finisher
        matrix, layers_index_reverse, branchs_end, branchs_counter = self.finisher_applier(
            matrix, layers_index_reverse,
            branchs_end, branchs_counter)

        self._matrix_updated = True
        self._matrix_mutated = matrix

        self._layers_index_reverse_updated = True
        self._layers_index_reverse_mutated = layers_index_reverse

    def freeze_state(self):
        for mutation in self.mutations_pool:
            if mutation.config.get('state', None) == 'broken':
                mutation.config['state'] = None
        # apply mutations
        self._matrix, self._layers_index_reverse, self.branchs_end, self.branchs_counter = self.mutations_applier(
            self._matrix, self._layers_index_reverse,
            self.branchs_end, self.branchs_counter)

        self.mutations_pool = []


    @property
    def matrix(self):
        """
        Return matrix with mutations
        """
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

    def dump(self):
        matrix = copy.deepcopy(self._matrix)
        matrix_mutated = copy.deepcopy(self._matrix_mutated)

        layers_index_reverse = {key: value.dump() for key, value in self._layers_index_reverse.items()}
        layers_index_reverse_mutated = {key: value.dump() for key, value in self._layers_index_reverse_mutated.items()}

        mutations_list = [i.dump() for i in self.mutations_pool]

        finisher = self._finisher.dump()
        branchs_end = self.branchs_end
        branchs_count = self.branchs_counter

        buffer = {}
        buffer['matrix'] = matrix
        buffer['matrix_mutated'] = matrix_mutated
        buffer['layers_index_reverse'] = layers_index_reverse
        buffer['layers_index_reverse_mutated'] = layers_index_reverse_mutated
        buffer['mutations_list'] = mutations_list
        buffer['finisher'] = finisher
        buffer['branchs_end'] = branchs_end
        buffer['branchs_count'] = branchs_count

        return buffer

    def load(self, data_load, distribution):
        self._matrix = np.array(data_load['matrix'])
        self._matrix_mutated = np.array(data_load['matrix_mutated'])
        self._layers_index_reverse = {key: Layer(value['layer_type'], distribution, None, None, None, value) for key, value in data_load['layers_index_reverse'].items()}
        self._layers_index_reverse_mutated = {key: Layer(value['layer_type'], distribution, None, None, None, value) for key, value in data_load['layers_index_reverse_mutated'].items()}

        self._layers_index_reverse = {int(key): value for key, value in self._layers_index_reverse.items()}
        self._layers_index_reverse_mutated = {int(key): value for key, value in self._layers_index_reverse_mutated.items()}

        self.mutations_pool = [MutationInjector(None, None, None, distribution, None, None, i) for i in data_load['mutations_list']]
        self._finisher = Layer(data_load['finisher']['layer_type'], distribution, None, None, None, data_load['finisher'])

        self.branchs_end = data_load['branchs_end']
        self.branchs_end = {int(key): value for key, value in self.branchs_end.items()}

        self.branchs_counter = data_load['branchs_count']


class StructureText(Structure):
    def __init__(self, root, embedding, finisher):
        """
        Initialize the architecture of the individual with textual data
        Can used in case of pure text as the input
        If own embedding - use pure Structure or 'general' as a type of data in Evolution

        Args:
            root {instance of the Layer} - input layer type
            embedding {instance of the Layer} - embedding layer type
        """
        super().__init__(root, finisher)

        self._matrix = np.zeros((2, 2))

        # add root layer - Input layer
        self._matrix, self._layers_index_reverse, root_index = self._register_new_layer(
            self._matrix,
            self._layers_index_reverse,
            root)

        # add embedding layer
        self._matrix, self._layers_index_reverse, embedding_index = self._register_new_layer(
            self._matrix,
            self._layers_index_reverse,
            embedding)

        self._matrix[root_index, embedding_index] = 1
        self._matrix_pure = self._matrix[:]

        self.branchs_end[1] = embedding_index


class StructureImage(Structure):
    def __init__(self, root, finisher):
        super().__init__(root, finisher)

        self._matrix = np.zeros((0, 0))
        self._matrix_pure = self._matrix[:]
        self._matrix, self._layers_index_reverse, root_index = self._register_new_layer(
            self._matrix,
            self._layers_index_reverse,
            root)

        self.branchs_end[1] = root_index
