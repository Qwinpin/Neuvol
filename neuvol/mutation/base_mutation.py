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
import math
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
        # TODO: external probabilities for each dice
        merger_dice = _probability_from_branched(individ)

        # branches, which was merged should not be splitted or grown after
        branches_exception = []
        while merger_dice:
            print('M')
            # TODO: generate distribution according to results of epochs
            branches_to_merge_number = np.random.randint(2, len(individ.branchs_end.keys()) + 1) if len(individ.branchs_end.keys()) >= 2 else 0
            branches_to_merge = np.random.choice(list(individ.branchs_end.keys()), branches_to_merge_number, replace=False)
            branches_to_merge = [i for i in branches_to_merge if i not in branches_exception]

            if len(branches_to_merge) < 2:
                break

            new_tail = Layer(Distribution.layer())

            individ.merge_branches(new_tail, branches_to_merge)

            merger_dice = _probability_from_branched(individ)

            branches_exception.append(branches_to_merge[0])

        # for each branch now we need decide split or not
        free_branches = [i for i in individ.branchs_end.keys() if i not in branches_exception]

        for branch in free_branches:
            print(branch)

            split_dice = 1 - _probability_from_branched(individ)

            if split_dice:
                print('split')
                number_of_splits = np.random.choice([2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.1])
                new_tails = [Layer(Distribution.layer()) for _ in range(number_of_splits)]

                individ.split_branch(new_tails, branch=branch)

            else:
                new_tail = Layer(Distribution.layer())

                individ.add_layer(new_tail, branch)

        return True


def _probability_from_branched(individ):
    number_of_branches = len(individ.branchs_end.keys())

    if number_of_branches > 1:
        probability = 1 - 1 / math.log(number_of_branches, 1.5)
        dice = np.random.choice([0, 1], p=[1 - probability, probability])
    else:
        dice = 0

    return dice
