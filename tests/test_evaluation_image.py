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
import copy
import random
import string
import unittest

import numpy as np

from neuvol.architecture import cradle
from neuvol.evaluation import Evaluator
from neuvol.layer.block import Block


class TestEvaluationImage(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('Hello, we start our tests right now\n')
        options = {'classes': 2, 'shape': (42, 42, 3,), 'depth': 8}
        self.individ = cradle(0, data_type='image', **options)
        self.individ.architecture = build_test_architecture(options)

        self.x = np.zeros((100, 42, 42, 3))
        self.y = np.random.randint(0, 2, size=(100))

        self.evaluator = Evaluator(self.x, self.y, kfold_number=1)

        print('We created one individ. Its name - {}'.format(self.individ))

    def test_evaluation_initialization(self):
        self.assertIsInstance(self.evaluator, Evaluator)

    def test_evaluation_fit(self):
        result = self.evaluator.fit(self.individ)
        self.assertGreaterEqual(result, 0.0)

    def test_evaluation_fitness_measure_error(self):
        self.evaluator.fitness_measure = 'magic'
        with self.assertRaises(TypeError):
            self.evaluator.fit(self.individ)

    def test_evaluation_fit_tensor_error(self):
        tmp = copy.deepcopy(self.individ)

        tmp2 = tmp.architecture
        tmp2.append(Block('cnn'))

        tmp.architecture = tmp

        with self.assertRaises(ArithmeticError):
            self.evaluator.fit(tmp)


def build_test_architecture(options):
    """
    Create simple architecture to avoid random initialization problem
    """
    tmp_architecture = []
    tmp_architecture.append(Block('input', layers_number=1, **options))
    tmp_architecture.append(Block('cnn2'))
    tmp_architecture.append(Block('last_dense', layers_number=1, **options))

    return tmp_architecture
