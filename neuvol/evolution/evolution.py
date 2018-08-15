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
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from ..architecture import cradle
from ..probabilty_pool import Distribution


class Evolution():
    """
    Simple class that performs evolution process
    """
    def __init__(
            self,
            stages,
            population_size,
            evaluator,
            mutator,
            crosser,
            data_type='text',
            task_type='classification',
            freeze=None,
            active_distribution=True,
            **kwargs):
        self._stages = stages
        self._population_size = population_size
        self._evaluator = evaluator
        self._mutator = mutator
        self._crosser = crosser
        self._data_type = data_type
        self._task_type = task_type
        self._freeze = freeze
        self._active_distribution = active_distribution
        self._options = kwargs

        self._population = []
        self._mutation_pool_size = 0.2
        self._mortality_rate = 0.2
        self._current_stage = 0

        if self._data_type == 'text':
            Distribution.set_layer_status('cnn2', active=False)
        elif self._data_type == 'image':
            Distribution.set_layer_status('cnn2', active=True)
            Distribution.set_layer_status('lstm', active=False)
            Distribution.set_layer_status('bi', active=False)
            Distribution.set_layer_status('max_pool', active=False)

        self._create_population()

    def _create_population(self):
        """
        First individ initialisation
        """
        for _ in range(self._population_size):
            self._population.append(
                cradle(0, self._data_type, self._task_type, freeze=self._freeze, **self._options))

    def mutation_step(self):
        """
        Mutate randomly chosen individs
        """
        for _ in range(int(self._mutation_pool_size * self._population_size)):
            index = int(np.random.randint(0, len(self._population)))
            self._population[index] = self._mutator.mutate(self._population[index], self._current_stage)

    def step(self):
        """
        Perform one step of evolution, that consists of evaluation and death
        """
        for network in self._population:
            try:
                network.result = self._evaluator.fit(network)
            except:
                # sorry, but here i dont care about type of exception
                network.result = 0.0

        best_individs = sorted(self._population, key=lambda individ: (-1) * individ.result)
        self._population = best_individs[:int(-self._mortality_rate * self._population_size)]

        self._current_stage += 1

    def crossing_step(self):
        """
        Cross two individs and create new one
        """
        for _ in range(self._population_size - len(self._population)):
            if np.random.choice([0, 1]):
                index_father = int(np.random.randint(0, len(self._population)))
                index_mother = int(np.random.randint(0, len(self._population)))

                new_individ = self._crosser.cross(
                    deepcopy(self._population[index_father]),
                    deepcopy(self._population[index_mother]), self._current_stage)

                self._population.append(new_individ)

            else:
                self._population.append(
                    cradle(0, self._data_type, self._task_type, freeze=self._freeze, **self._options))

    def _population_probability(self):
        for individ in self._population[:3]:
            Distribution.parse_architecture(individ)

    def cultivate(self):
        """
        Perform all evolutional steps
        """
        tmp = self._current_stage

        for i in range(tmp, self._stages):
            print('\nStage #{} of {}\n'.format(i, self._stages))

            self.mutation_step()
            self.step()
            if self._active_distribution:
                self._population_probability()
            self.crossing_step()

    @property
    def population(self, n=5):
        """
        Get the architecture of n individs
        """
        if n > self._population_size:
            n = self._population_size

        for i in range(n):
            print('Name: ', self._population[i], '\n')
            for block in self._population[i].architecture:
                print('Block: \t ', block.type)
                print('Config: \t', block.config_all, '\n')

            print('\n\n')

    def population_raw_individ(self):
        return self._population

    @property
    def stages(self):
        return self._stages

    @stages.setter
    def stages(self, stages):
        self._stages = stages
