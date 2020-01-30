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
import unittest

import numpy as np

from neuvol.crossing import Crosser
from neuvol.evaluation import Evaluator
from neuvol.evolution import Evolution
from neuvol.mutation import Mutator


class TestEvolutionImage(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('Hello, we start our tests right now\n')
        options = {'classes': 2, 'shape': (42, 42, 3,), 'depth': 1}

        self.x = np.zeros((100, 42, 42, 3))
        self.y = np.random.randint(0, 2, size=(100))

        self.evaluator = Evaluator(self.x, self.y, kfold_number=1)
        self.evaluator.create_tokens = False

        self.mutator = Mutator()
        self.crosser = Crosser()

        self.evolution = Evolution(1, self.evaluator, self.mutator, self.crosser, 10, 'image', **options)

    def test_evolution_initialization(self):
        self.assertIsInstance(self.evolution, Evolution)

    def test_evolution_mutation(self):
        self.evolution.mutation_step()
        self.assertEqual(10, len(self.evolution.population()))

    def test_evolution_crossing(self):
        self.evolution.crossing_step()
        self.assertEqual(10, len(self.evolution.population()))
