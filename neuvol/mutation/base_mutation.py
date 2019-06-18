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

from ..probabilty_pool import Distribution
from ..layer import Layer


class MutatorBase:
    """
    Mutator class for textual data
    """

    @staticmethod
    def mutate(individ):
        """
        Mutate individ
        """
        pass

    @staticmethod
    def grown(individ):
        number_of_branches = individ.architecture.branch_count

        merger_dice = 0 if number_of_branches == 1 else np.random.choice([0, 1])

        # branches, which was merged should not be splitted or grown after
        branches_exception = []
        while merger_dice is True:
            # TODO: generate distribution according to results of epochs
            branches_to_merge_number = np.random.randint(len(individ.branchs_end.keys()))
            branches_to_merge = np.random.choice([i + 1 for i in range(branches_to_merge_number)])
            branches_to_merge = [i for i in branches_to_merge if i not in branches_exception]

            if len(branches_to_merge) < 2:
                break

            new_tail = Layer(Distribution.layer())

            individ.merge_branches(new_tail, branches_to_merge)

            number_of_branches = individ.architecture.branch_count
            merger_dice = 0 if len(individ.branchs_end.keys()) == 1 else np.random.choice([0, 1])

            branches_exception.append(branches_to_merge[0])

        # for each branch now we need decide split or not
        free_branches = [i for i in individ.branchs_end.keys() if i not in branches_exception]

        for branch in free_branches:
            new_tail = Layer(Distribution.layer())

            individ.add_layer(new_tail, branch)

        return True
