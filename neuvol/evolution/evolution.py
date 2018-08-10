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
        self.stages = stages
        self.population_size = population_size
        self.evaluator = evaluator
        self.mutator = mutator
        self.crosser = crosser
        self.data_type = data_type
        self.task_type = task_type
        self.freeze = freeze
        self.active_distribution = active_distribution
        self.options = kwargs

        self.population = []
        self.mutation_pool_size = 0.2
        self.mortality_rate = 0.2
        self.current_stage = 0

        if self.data_type == 'text':
            Distribution.set_layer_status('cnn2', active=False)
        elif self.data_type == 'image':
            Distribution.set_layer_status('cnn2', active=True)
            Distribution.set_layer_status('lstm', active=False)
            Distribution.set_layer_status('bi', active=False)
            Distribution.set_layer_status('max_pool', active=False)

        self._create_population()

    def _create_population(self):
        """
        First individ initialisation
        """
        for _ in range(self.population_size):
            self.population.append(
                cradle(0, self.data_type, self.task_type, freeze=self.freeze, **self.options))

    def mutation_step(self):
        """
        Mutate randomly chosen individs
        """
        for _ in range(int(self.mutation_pool_size * self.population_size)):
            index = int(np.random.randint(0, len(self.population)))
            self.population[index] = self.mutator.mutate(self.population[index], self.current_stage)

    def step(self):
        """
        Perform one step of evolution, that consists of evaluation and death
        """
        for network in self.population:
            try:
                network.result = self.evaluator.fit(network)
            except:
                network.result = 0.0

        best_individs = sorted(self.population, key=lambda individ: (-1) * individ.result)
        self.population = best_individs[:int(-self.mortality_rate * self.population_size)]

    def crossing_step(self):
        """
        Cross two individs and create new one
        """
        for _ in range(self.population_size - len(self.population)):
            if np.random.choice([0, 1]):
                index_father = int(np.random.randint(0, len(self.population)))
                index_mother = int(np.random.randint(0, len(self.population)))

                new_individ = self.crosser.cross(
                    deepcopy(self.population[index_father]),
                    deepcopy(self.population[index_mother]), self.current_stage)

                self.population.append(new_individ)

            else:
                self.population.append(
                    cradle(0, self.data_type, self.task_type, freeze=self.freeze, **self.options))


    def population_probability(self):
        for individ in self.population[:3]:
            Distribution.parse_architecture(individ)

    def cultivate(self):
        """
        Perform all evolutional steps
        """
        for i in tqdm(range(self.stages)):
            print('\nStage #{}\n'.format(i))

            self.current_stage += i
            self.mutation_step()
            self.step()
            if self.active_distribution:
                self.population_probability()
            self.crossing_step()
