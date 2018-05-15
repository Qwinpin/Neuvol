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

from .. import architecture


class Evolution():
    def __init__(self, stages, population_size, evaluator, data_type='text', task_type='classification', freeze=None,
                 **kwargs):
        self.stages = stages
        self.population_size = population_size
        self.evaluator = evaluator
        self.data_type = data_type
        self.task_type = task_type
        self.freeze = freeze
        self.options = kwargs

        self.population = []
        self.mutation_pool_size = 0.2
        self.mortality_rate = 0.2
        self.current_stage = 0

        self._create_population()

    def _create_population(self):
        for _ in range(self.population_size):
            self.population.append(
                architecture.Individ(0, self.data_type, self.task_type, freeze=self.freeze, **self.options))

    def mutation_step(self):
        for _ in range(int(self.mutation_pool_size * self.population_size)):
            index = int(np.random.randint(0, len(self.population)))
            self.population[index].mutation(self.current_stage)

    def step(self):
        for network in self.population:
            network.result = self.evaluator.fit(network)

        best_individs = sorted(self.population, key=lambda individ: (-1) * individ.result)
        self.population = best_individs[:int(-self.mortality_rate * self.population_size)]

    def crossing_step(self):
        for _ in range(self.population_size - len(self.population)):
            index_father = int(np.random.randint(0, len(self.population)))
            index_mother = int(np.random.randint(0, len(self.population)))

            new_individ = self.population[index_father].crossing(
                self.population[index_mother],
                self.current_stage)

            self.population.append(new_individ)

    def cultivate(self):
        for i in range(self.stages):
            print('\nStage #{}\n'.format(i))
            self.current_stage = i
            self.mutation_step()
            self.step()
            self.crossing_step()
