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
from ...layer.reshaper import calculate_shape, reshaper, merger


class Structure:
    def __init__(self, root):
        self.current_depth = 0
        self.current_width = 1
        self.branch_count = 1
        pass

    def add_layer(self, layer, branch):
        pass

    def merge_branch(self, left_branch, right_branch):
        pass

    def split_branch(self, root, left, right):
        pass


class StructureText(Structure):
    def __init__(self, root, embedding):
        self.current_depth = 1
        self.current_width = 1
        self.branch_count = 1

        self.tree = {'root': ['embedding']}
        self.branchs_end = {1: 'embedding'}


        embedding.config['rank'], embedding.config['shape'] = calculate_shape(root, embedding)
        self.layers = {'root': root, 'embedding': embedding}

        self.current_depth += 1

    def add_layer(self, layer, branch, branch_out=None):
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
        left_to = self.branchs_end[left_branch]
        right_to = self.branchs_end[right_branch]

        left_to_object = self.layers[left_to]
        right_to_object = self.layers[right_to]

        left_modifier_reshaper, left_shape_modifier = merger(left_to_object, layer)

        self.add_layer(left_shape_modifier, left_to_object)
        self.add_layer(right_shape_modifier, right_to_object)

        self.add_layer(left_modifier_reshaper, left_to_object)
        self.add_layer(right_modifier_reshaper, right_to_object)

        new_name = '{}_{}'.format(self.current_depth, left_branch)

        if self.tree.get(left_to) is None:
            self.tree[left_to] = []

        if self.tree.get(right_to) is None:
            self.tree[right_to] = []

        self.tree[left_to].append(new_name)
        self.tree[right_to].append(new_name)

        self.branch_count -= 1
        del self.branchs_end[right_branch]
        self.branchs_end[left_branch] = new_name
        self.layers[new_name] = layer

        self.current_depth += 1
        self.current_width -= 1

    def split_branch(self, left_layer, right_layer, branch):
        # call simple add for each branch
        self.add_layer(right_layer, branch, branch_out=self.branch_count + 10)
        self.current_depth -= 1
        self.add_layer(left_layer, branch)

        self.current_depth -= 1  # hm, we use two add layer operation at onces, we need only one (+1) to the depth
