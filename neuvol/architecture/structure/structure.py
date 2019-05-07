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
from ...layer.reshaper import calculate_shape, reshaper, merger, merger_mass


class Structure:
    def __init__(self, root):
        self.current_depth = 0
        self.current_width = 1
        self.branch_count = 1

        self.tree = {}
        self.branchs_end = {}
        self.layers = {}

    def add_layer(self, layer, branch, branch_out=None):
        """
        Add layer to the last layer of the branch

        Args:
            layer (instance of the Layer)
            branch (int) - number of the branch to be connected to
            branch_out (int) - number of the branch after this new layer: if branch is splitted
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
            layer.config['rank'], layer.config['shape'] = calculate_shape(add_to_object, layer)
        
        modifier_reshaper = reshaper(add_to_object, layer)
        if modifier_reshaper is not None:
            # now we want to connect new layer through the reshaper
            add_to = self.add_layer(modifier_reshaper, branch, branch_out=branch_out)
            add_to_object = self.layers[add_to]
            layer.config['rank'], layer.config['shape'] = calculate_shape(add_to_object, layer)

        # if not None - we want to create a new branch
        if branch_out is not None:
            branch = branch_out

        new_name = '{}_{}'.format(self.current_depth, branch)

        if self.tree.get(add_to) is None:
            self.tree[add_to] = []

        self.tree[add_to].append(new_name)
        self.branchs_end[branch] = new_name
        self.layers[new_name] = layer

        self.current_depth += 1

        return new_name

    def merge_branch(self, layer, left_branch, right_branch):
        """
        Merge two branchs to a new layer
        
        Args:
            layer (instance of the Layer)
            left_branch (int)
            right_branch (int)
        """
        left_to = self.branchs_end[left_branch]
        right_to = self.branchs_end[right_branch]

        left_to_object = self.layers[left_to]
        right_to_object = self.layers[right_to]

        modifier, shape_modifier = merger(left_to_object, right_to_object)

        if shape_modifier is not None:
            left_to = self.add_layer(shape_modifier, left_branch)
            right_to = self.add_layer(shape_modifier, right_branch)
        
        if self.tree.get(left_to) is None:
            self.tree[left_to] = []

        if self.tree.get(right_to) is None:
            self.tree[right_to] = []

        modifier_name = '{}_{}'.format(self.current_depth, left_branch)
        self.tree[left_to].append(modifier_name)
        self.tree[right_to].append(modifier_name)

        self.branchs_end[left_branch] = modifier_name
        self.layers[modifier_name] = modifier
        self.current_depth += 1

        del self.branchs_end[right_branch]
        self.branch_count -= 1

        new_name = self.add_layer(layer, left_branch)
        return new_name

    def merge_branches(self, layer, branches=None):
        add_to = [self.branchs_end[branch] for branch in branches]
        add_to_objects = [self.layers[to] for to in add_to]

        modifier, shape_modifier = merger_mass(add_to_objects)

        if shape_modifier is not None:
            add_to = [self.add_layer(shape_modifier, branch) for branch in branches]

        for to in add_to:
            if self.tree.get(to) is None:
                self.tree[to] = []

        modifier_name =  'm{}_{}'.format(self.current_depth, branches[0])

        for to in add_to:
            self.tree[to].append(modifier_name)

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
            left_layer (instance of the Layer) - layer, which forms a left branch
            right_layer (instance of the Layer) - layer, which forms a right branch
            branch (int) - branch, which should be splitted
        """
        # call simple add for each branch
        self.add_layer(right_layer, branch, branch_out=self.branch_count + 1)
        self.add_layer(left_layer, branch)

        self.branch_count += 1


class StructureText(Structure):
    def __init__(self, root, embedding):
        """
        Initialize the architecture of the individual
        
        Args:
            root (instance of the Layer) - input layer type
            embedding (instance of the Layer) - embedding layer type
        """
        super().__init__(root)

        self.tree['root'] = ['embedding']
        self.branchs_end[1] = 'embedding'

        embedding.config['rank'], embedding.config['shape'] = calculate_shape(root, embedding)
        self.layers['root'] = root
        self.layers['embedding'] = embedding

        self.current_depth += 1


class StructureImage(Structure):
    def __init__(self, root):
        super().__init__(root)

        self.tree['root']
        self.branchs_end[1] = 'root'

        self.layers['root'] = root
