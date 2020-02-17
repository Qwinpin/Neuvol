# Copyright 2020 Timur Sokhin.
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
from ..layer.capsule_layer import detect_best_combination, structure_parser, remove_duplicated_branches
from ..mutation import MutationInjector
from ..utils import parameters_copy


class Crosser:
    @parameters_copy
    def cross(self, individ1, individ2, size=5):
        # detect branch in individ1, which will replace branch in individ2
        ind1_complex = structure_parser(individ1.architecture, size)
        ind1_complex = [detect_best_combination(i) for i in ind1_complex]
        ind1_complex = [remove_duplicated_branches(new_chain) for new_chain in ind1_complex if new_chain]
        if len(ind1_complex) == 0:
            return None
        ind1_complex = [i for i in ind1_complex if i][0]

        # detect branch in individ2, which will be dropped in invidid2
        ind2_complex = structure_parser(individ2.architecture, size)
        ind2_complex = [detect_best_combination(i) for i in ind2_complex]
        ind2_complex = [remove_duplicated_branches(new_chain) for new_chain in ind2_complex if new_chain]
        if len(ind2_complex) == 0:
            return None
        ind2_complex = [i for i in ind2_complex if i][0]

        self.cut_branch(individ2, ind2_complex)
        self.inject_branch(individ2, individ1, ind1_complex, ind2_complex[0][0], ind2_complex[0][-1])

        return individ2

    def cut_branch(self, individ, branch):
        for chain in branch:
            for index in [chain[i: i + 2] for i in range(len(chain[::2]) + 1)]:
                if len(index) != 2:
                    return False
                remove_connection_mutation = MutationInjector(None, None, None, None)
                remove_connection_mutation.mutation_type = 'remove_connection'
                remove_connection_mutation.after_layer_index = index[0]
                remove_connection_mutation.before_layer_index = index[1]
                individ.add_mutation(remove_connection_mutation)

    def inject_branch(self, individ, individ_donor, branch, from_index, to_index):
        tmp_map = {}
        # layers_reverse is used to get new layer index after mutation
        # [:-1] - because of last layer, which is temporary finisher
        initial_layers_reverse = set(list(individ.layers_index_reverse.keys())[:-1])
        for chain in branch:
            # we go through the tail to the head
            for index in chain[::-1]:
                # if this index was added before - just add required connection withour layer duplication
                if tmp_map.get(index, None) is None:
    #                 remove_connection_mutation = MutationInjector(None, None, None, None)
    #                 remove_connection_mutation.mutation_type = 'remove_connection'
    #                 remove_connection_mutation.after_layer_index = from_index
    #                 remove_connection_mutation.before_layer_index = to_index

    #                 individ.add_mutation(remove_connection_mutation)
                    inject_layer_mutation = MutationInjector(None, None, None, None)
                    inject_layer_mutation.mutation_type = 'inject_layer'
                    inject_layer_mutation.layer = individ_donor.layers_index_reverse[index]
                    inject_layer_mutation.after_layer_index = from_index
                    inject_layer_mutation.before_layer_index = to_index

                    individ.add_mutation(inject_layer_mutation)
                    new_layers_reverse = set(list(individ.layers_index_reverse.keys())[:-1]) - initial_layers_reverse
                    to_index = list(new_layers_reverse)[0]
                    initial_layers_reverse = set(list(individ.layers_index_reverse.keys())[:-1])

                    tmp_map[index] = to_index
                else:
                    add_connection_mutation = MutationInjector(None, None, None, None)
                    add_connection_mutation.mutation_type = 'add_connection'
                    add_connection_mutation.after_layer_index = tmp_map[index]
                    add_connection_mutation.before_layer_index = to_index

                    individ.add_mutation(add_connection_mutation)
