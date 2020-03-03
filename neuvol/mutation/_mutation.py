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
from .mutation_interface import mutation
from ..constants import EVENT


class Mutator():
    """
    Class for mutation performing
    """
    def __init__(self):
        pass

    def mutate(self, network, stage):
        """
        Darwin was right. Change some part of individ with some probability
        """
        network.history = EVENT('Mutation', stage)

        return mutation(network.data_type).mutate(network)
