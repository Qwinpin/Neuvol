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
import os

import numpy as np

from ..individs import cradle
from ..probabilty_pool import Distribution
from ..utils import dump, load


class Evolution():
    """
    Simple class that performs evolution process
    """
    def __init__(
            self,
            stages,
            evaluator,
            mutator,
            crosser,
            population_size=10,
            data_type='text',
            task_type='classification',
            freeze=None,
            active_distribution=True,
            loaded=False,
            multiple_gpu=False,
            vizualize=False,
            **kwargs):
        # TODO: methods of changing the parameters
        self._evaluator = evaluator
        self._mutator = mutator
        self._crosser = crosser
        self._stages = stages
        self._population_size = population_size

        self._data_type = data_type
        self._task_type = task_type
        self._freeze = freeze
        self._active_distribution = active_distribution
        self._options = kwargs

        self._population = []
        self._mutation_pool_size = 0.2
        self._mortality_rate = 0.2
        self._current_stage = 1

        self._viz_data = {}
        self._viz_data['population'] = []

        if self._data_type == 'text':
            Distribution.set_layer_status('cnn2', active=False)
            Distribution.set_layer_status('max_pool2', active=False)
            Distribution.set_layer_status('lstm', active=True)
            Distribution.set_layer_status('bi', active=True)
            Distribution.set_layer_status('max_pool', active=True)
            Distribution.set_layer_status('cnn', active=True)

        elif self._data_type == 'image':
            Distribution.set_layer_status('cnn2', active=True)
            Distribution.set_layer_status('max_pool2', active=True)
            Distribution.set_layer_status('lstm', active=False)
            Distribution.set_layer_status('bi', active=False)
            Distribution.set_layer_status('max_pool', active=False)
            Distribution.set_layer_status('cnn', active=False)

        if not loaded:
            self._create_population()

    def _create_population(self):
        """
        First individ initialisation
        """
        for _ in range(self._population_size):
            self._population.append(
                cradle(self._current_stage, self._data_type, self._task_type, freeze=self._freeze, **self._options))

    def mutation_step(self):
        """
        Mutate randomly chosen individs
        """
        # mutate only previous elements in the population
        for network in range(round(self._population_size * (1 - self._mortality_rate))):
            if np.random.randint(0, 1) > 0.3:
                # TODO: more accurate error handling
                self._population[network] = self._mutator.mutate(self._population[network], self._current_stage + 1)

                # set result as -1 to retrain net
                self._population[network].result = -1.0

    def step(self):
        """
        Perform one step of evolution, that consists of evaluation and death
        """
        for network in self._population:
            network.stage = self._current_stage

            # train net only if we did not do this before
            if network.result == -1:
                try:
                    network.result = self._evaluator.fit(network)
                # NOTE: maybe ArithmeticError ?
                except Exception as e:
                    # sorry, but here i dont care about type of exception
                    print(e)
                    network.result = 0.0

        self.viz()

        best_individs = sorted(self._population, key=lambda individ: (-1) * individ.result)
        self._population = best_individs[:int(self._population_size // (self._mortality_rate * 10))]

    def crossing_step(self):
        """
        Cross two individs and create new one
        """
        old_population_size = self._population_size - len(self._population)
        for _ in range(old_population_size):
            if np.random.choice([0, 1]):
                index_father = int(np.random.randint(0, old_population_size))
                index_mother = int(np.random.randint(0, old_population_size))

                # TODO: more accurate error handling
                new_individ = self._crosser.cross(
                    deepcopy(self._population[index_father]),
                    deepcopy(self._population[index_mother]), self._current_stage + 1)

                self._population.append(new_individ)

            else:
                self._population.append(
                    cradle(self._current_stage, self._data_type, self._task_type, freeze=self._freeze, **self._options))

    def _population_probability(self):
        for individ in self._population[:3]:
            Distribution.parse_architecture(individ)

    def cultivate(self):
        """
        Perform all evolutional steps
        """
        tmp = self._stages + self._current_stage

        for i in range(self._stages):
            print('\nStage #{} of {}\n'.format(self._current_stage, tmp - 1))

            if self._current_stage == 0:
                self.step()
                if self._active_distribution:
                    self._population_probability()
            else:
                self.crossing_step()
                self.mutation_step()
                self.step()
                if self._active_distribution:
                    self._population_probability()

            print('\nBest result: {}\n'.format(self._population[0].result))

            self._current_stage += 1
            self.dump('tmp.json')

    def save(self):
        """
        Serialize object for further dump
        """
        serial = dict()
        serial['stages'] = self._stages
        serial['population_size'] = self._population_size
        serial['data_type'] = self._data_type
        serial['task_type'] = self._task_type
        serial['freeze'] = self._freeze
        serial['active_distribution'] = self._active_distribution
        serial['options'] = self._options

        serial['population'] = [individ.save() for individ in self._population]
        serial['mutation_pool_size'] = self._mutation_pool_size
        serial['mortality_rate'] = self._mortality_rate
        serial['current_stage'] = self._current_stage

        return serial

    def dump(self, path):
        """
        Dump evolution states including individs
        """
        dump(self.save(), path)

    def viz(self):
        """
        Serialize the whole story of evolution for further visualization
        """
        if os.path.isfile('./viz_data.json'):
            self._viz_data = load('viz_data.json')
        else:
            self._viz_data = {}
            self._viz_data['population'] = []

        for network in self._population:
            tmp = deepcopy(network)

            # if individ was created rigth now - we dont remove parents to connect them
            # if individ was created early - set parents as None to avoid crossconnections
            if tmp.history[0].type == 'Birth':
                if len(tmp.history) != 1:
                    tmp._parents = None
                else:
                    tmp._parents[0]._name = '_'.join([tmp._parents[0]._name, str(self._current_stage - 1)])
                    tmp._parents[1]._name = '_'.join([tmp._parents[1]._name, str(self._current_stage - 1)])

            if tmp.history[0].stage < self._current_stage:
                tmp.options['previous_train'] = '_'.join([tmp._name, str(self._current_stage - 1)])
            else:
                tmp.options['previous_train'] = None

            tmp._name = '_'.join([tmp._name, str(self._current_stage)])
            self._viz_data['population'].append(tmp.save())

        self._viz_data['current_stage'] = self._current_stage

        dump(self._viz_data, './viz_data.json')
        self._viz_data = None

    @staticmethod
    def load(file_name, evaluator, mutator, crosser):
        """
        Load evolution state from the file and additional information: evaluator, mutator and crosser
        """
        serial = load(file_name)
        evolution = Evolution(serial['stages'], evaluator, mutator, crosser, loaded=True)

        evolution._stages = serial['stages']
        evolution._population_size = serial['population_size']
        evolution._data_type = serial['data_type']
        evolution._task_type = serial['task_type']
        evolution._freeze = serial['freeze']
        evolution._active_distribution = serial['active_distribution']
        evolution._options = serial['options']
        evolution._mutation_pool_size = serial['mutation_pool_size']
        evolution._mortality_rate = serial['mortality_rate']
        evolution._current_stage = serial['current_stage']

        # in order to avoid overhead with creation and replacing of parameters, we use classmethod.
        # it allow us to create only one object, set data type and then use it as a cradle
        tempory_individ = cradle(None, evolution._data_type)
        evolution._population = [tempory_individ.load(data) for data in serial['population']]

        return evolution

    def population(self):
        """
        Get the whole population itself
        """
        return self._population

    @property
    def stages(self):
        """
        Number of stages
        """
        return self._stages

    @stages.setter
    def stages(self, stages):
        self._stages = stages
