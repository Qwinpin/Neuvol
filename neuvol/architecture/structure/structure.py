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
from ...layer.reshaper import calculate_shape, reshaper


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

        self.tree = {'root': 'embedding'}
        self.branchs_end = {1: 'embedding'}


        embedding.config['rank'], embedding.config['shape'] = calculate_shape(root, embedding)
        self.layers = {'root': root, 'embedding': embedding}

        self.current_depth += 1

    def add_layer(self, layer, branch):
        if layer is not None:
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
                add_to = self.add_layer(modifier_reshaper, branch)
                add_to_object = self.layers[add_to]
                layer.config['rank'], layer.config['shape'] = calculate_shape(add_to_object, layer)

            self.tree[add_to] = '{}_{}'.format(self.current_depth, branch)
            self.branchs_end[branch] = self.tree[add_to]
            self.layers[self.tree[add_to]] = layer

            self.current_depth += 1

            return self.tree[add_to]

    def merge_branch(self, layer, left_branch, right_branch):
        left_to = self.branchs_end[left_branch]
        right_to = self.branchs_end[right_branch]

        left_to_object = self.layers[left_to]
        right_to_object = self.layers[right_to]

        left_modifier_reshaper = reshaper(left_to_object, layer)
        right_modifier_reshaper = reshaper(right_to_object, layer)

        self.add_layer(left_modifier_reshaper, left_to_object)
        self.add_layer(right_modifier_reshaper, right_to_object)

        self.tree[left_to] = '{}_{}'.format(self.current_depth, left_branch)
        self.tree[right_to] = '{}_{}'.format(self.current_depth, left_branch)

        self.branch_count -= 1
        del self.branchs_end[right_branch]
        self.branchs_end[left_branch] = self.tree[left_to]
        self.layers[self.tree[left_to]] = layer

        self.current_depth += 1
        self.current_width -= 1

    def split_branch(self, left_layer, right_layer, branch):
        pass
