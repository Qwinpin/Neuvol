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

from neuvol.architecture import cradle
from neuvol.layer.block import Block


class TestArchitecture(unittest.TestCase):

    def setUp(self):
        print('Intitialization of the individ for tests\n')
        self.individ = cradle(0)
        print('Its name - {}'.format(self.individ))

    def test_architecture_tf_building(self):
        self.assertEqual(3, len(self.individ.init_tf_graph()))

    def test_architecture_block_structure(self):
        for block in self.individ.architecture:
            self.assertIsInstance(type(block), Block)

    def test_architecture_training_parameters(self):
        self.assertIsNotNone(self.individ.training_parameters)

    def test_architecture_data_parameters(self):
        self.assertIsNotNone(self.individ.data_processing)
    
    def test_architecture_layers_number(self):
        self.assertNotEqual(self.individ.layers_number, 0)

    def test_architecture_default_initial_parameters(self):
        self.assertIsNone(self.individ.options)

    def test_architecture_tf_building_error(self):
        tmp = cradle(0, task_type='magic')
        self.assertRaises(Exception('Unsupported task type'), tmp.init_tf_graph())
