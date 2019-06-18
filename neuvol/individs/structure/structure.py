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

from ...layer.reshaper import calculate_shape, reshaper, reshaper_shape, merger_mass


class Structure:
    def __init__(self, root):
        self.current_depth = 0
        self.branch_count = 1

        # dict form of architecture, which is used for initialization of tensor graph
        self.tree = {}

        # matrix form of architecture, which is used for mutations, crossing, etc
        self.matrix = None

        # current endings (layer) of each branch
        self.branchs_end = {}

        # layer dict layer_name: layer_object
        self.layers = {}

        # map layer_name: layer_index, monotonically increasing
        self.layers_indexes = {}
        # map layer_index: layer_name
        self.layers_indexes_reverse = {}
        self.layers_counter = 0

        # final layer of the whole network
        self.finisher = None

    def _register_new_layer(self, new_layer_name):
        """Add new layer to the indexers and increase the size of layers matrix

        Args:
            new_layer_name {str} - name of new layer
        """
        tmp_matrix = np.zeros((self.matrix.shape[0] + 1, self.matrix.shape[1] + 1))
        tmp_matrix[:self.matrix.shape[0], :self.matrix.shape[1]] = self.matrix

        self.matrix = tmp_matrix

        self.layers_indexes[new_layer_name] = len(self.layers_indexes)
        self.layers_indexes_reverse[len(self.layers_indexes_reverse)] = new_layer_name

    def add_layer(self, layer, branch, branch_out=None):
        """
        Add layer to the last layer of the branch

        Args:
            layer {instance of the Layer}
            branch {int} - number of the branch to be connected to
            branch_out {int} - number of the branch after this new layer: if branch is splitted
        """
        # branch_out used if you add new layer as a separate branch
        if layer is None:
            return None

        add_to = self.branchs_end[branch]
        add_to_object = self.layers[add_to]

        # check if new shape is known or not
        # for instance, shape is already known for embedding, input, reshape layers
        new_shape = layer.config.get('shape', None)
        if new_shape is None:
            layer.config['rank'], layer.config['shape'] = layer.calculate_rank(add_to_object), layer.calculate_shape(add_to_object)

        modifier_reshaper = reshaper(add_to_object, layer)
        if modifier_reshaper is not None:
            # now we want to connect new layer through the reshaper
            add_to = self.add_layer(modifier_reshaper, branch, branch_out=branch_out)
            add_to_object = self.layers[add_to]
            layer.config['rank'], layer.config['shape'] = layer.calculate_rank(add_to_object), layer.calculate_shape(add_to_object)

        # if not None - we want to create a new branch
        if branch_out is not None:
            branch = branch_out

        new_name = '{}_{}'.format(self.current_depth, branch)

        # add new layer to the tree form
        if self.tree.get(add_to) is None:
            self.tree[add_to] = []
        self.tree[add_to].append(new_name)

        # increase matrix shape and add new connection
        self._register_new_layer(new_name)
        self.matrix[self.layers_indexes[add_to], self.layers_indexes[new_name]] = 1

        # change branch ending
        self.branchs_end[branch] = new_name
        self.layers[new_name] = layer

        self.current_depth += 1
        self.finisher = new_name

        return new_name

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
        # now we prepare list if branch endings: names and objects itself
        add_to = [self.branchs_end[branch] for branch in branches]
        add_to_objects = [self.layers[to] for to in add_to]

        # calculate shapes if necessary
        modifier, shape_modifiers = merger_mass(add_to_objects)

        if shape_modifiers is not None:
            add_to = [self.add_layer(shape_modifiers[i], branch) for i, branch in enumerate(branches)]

        for to in add_to:
            if self.tree.get(to) is None:
                self.tree[to] = []

        modifier_name = 'm{}_{}_{}'.format(self.current_depth, branches[0], len(branches))

        self._register_new_layer(modifier_name)

        for to in add_to:
            self.tree[to].append(modifier_name)
            self.matrix[self.layers_indexes[to], self.layers_indexes[modifier_name]] = 1

        self.branchs_end[branches[0]] = modifier_name
        self.layers[modifier_name] = modifier
        self.current_depth += 1

        for branch in branches[1:]:
            del self.branchs_end[branch]
            self.branch_count -= 1

        new_name = self.add_layer(layer, branches[0])
        return new_name

    def split_branch(self, left_layer, right_layer, branch):
        """
        Split branch into two new branches

        Args:
            left_layer {instance of the Layer} - layer, which forms a left branch
            right_layer {instance of the Layer} - layer, which forms a right branch
            branch {int} - branch, which should be splitted
        """
        # call simple add for each branch
        self.add_layer(right_layer, branch, branch_out=self.branch_count + 1)
        self.add_layer(left_layer, branch)

        self.branch_count += 1

    def recalculate_shapes(self):
        self._recalculate_shapes(self.finisher)

    def _recalculate_shapes(self, heap=None):
        if heap is None:
            # if heap is undefined - start it from the known finisher
            target_index = self.layers_indexes[self.finisher]
        else:
            target_index = self.layers_indexes[heap]
            sources_indexes = np.where(self.matrix[:, target_index] == 1)

        target_object = self.layers[self.layers_indexes_reverse[target_index]]
        source_objects = [(int(source), self.layers[self.layers_indexes_reverse[int(source)]])
                          for source in sources_indexes[0]]

        # in case of merger layer
        if self.layers_indexes_reverse[target_index][0] == 'm':
            for source_index, source_object in source_objects:
                if source_object.config['shape'] is None:
                    source_object.config['shape'] = self._recalculate_shapes(self.layers_indexes_reverse[source_index])

            shapes = [item.config['shape'][1:] for i, item in source_objects]
            target_object.config['shape'] = (None, np.sum(shapes))

        else:
            for source_index, source_object in source_objects:
                if source_object.config['shape'] is None:
                    source_object.config['shape'] = self._recalculate_shapes(self.layers_indexes_reverse[source_index])

                if target_object.type == 'reshape':
                    difference_rank = source_object.config['rank'] - target_object.config['rank']

                    new_shape = reshaper_shape(difference=difference_rank, prev_layer=source_object)

                    target_object.config['shape'] = new_shape
                    target_object.config['target_shape'] = new_shape[1:]

                elif self.layers_indexes_reverse[target_index][0] != 'm':
                    target_object.config['rank'] = target_object.calculate_rank(source_object)
                    target_object.config['shape'] = target_object.calculate_shape(source_object)

        self.layers[self.layers_indexes_reverse[target_index]].config['shape'] = target_object.config['shape']
        return target_object.config['shape']


class StructureText(Structure):
    def __init__(self, root, embedding):
        """
        Initialize the architecture of the individual

        Args:
            root {instance of the Layer} - input layer type
            embedding {instance of the Layer} - embedding layer type
        """
        super().__init__(root)

        self.tree['root'] = ['embedding']
        self.branchs_end[1] = 'embedding'

        embedding.config['rank'], embedding.config['shape'] = embedding.calculate_rank(root), embedding.calculate_shape(root)
        self.layers['root'] = root
        self.layers['embedding'] = embedding

        self.matrix = np.zeros((2, 2))

        # add new layers to the indexes
        self._register_new_layer('root')
        self._register_new_layer('embedding')

        # using layers indexes we can fill the matrix of the graph
        self.matrix[self.layers_indexes['root'], self.layers_indexes['embedding']] = 1

        self.current_depth += 1


class StructureImage(Structure):
    def __init__(self, root):
        super().__init__(root)

        self.layers['root'] = root

        self.matrix = np.zeros((1, 1))

        # add new layers to the indexes
        self._register_new_layer('root')
