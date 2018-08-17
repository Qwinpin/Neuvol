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
from neuvol.architecture.individ_base import IndividBase
from neuvol.crossing.pairing_modules import peform_pairing
from neuvol.layer.block import Block


class TestArchitectureImage(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('Hello, we start our tests right now\n')
        options = {'classes': 10, 'shape': (32,), 'depth': 8}
        self.individ = cradle(0, data_type='image', **options)
        print('We created one individ. Its name - {}'.format(self.individ))

    def test_architecture_initialization_error(self):
        with self.assertRaises(KeyError):
            cradle(0)

    @unittest.expectedFailure
    def test_architecture_tf_building(self):
        self.assertEqual(3, len(self.individ.init_tf_graph()))

    def test_architecture_block_structure(self):
        for block in self.individ.architecture:
            self.assertIsInstance(block, Block)

    def test_architecture_training_parameters(self):
        self.assertIsNotNone(self.individ.training_parameters)

    def test_architecture_data_parameters(self):
        self.assertIsNotNone(self.individ.data_processing)

    def test_architecture_layers_number(self):
        self.assertNotEqual(self.individ.layers_number, 0)

    def test_architecture_default_initial_parameters(self):
        self.assertIsNotNone(self.individ.options)

    def test_architecture_tf_building_error(self):
        options = {'classes': 10, 'shape': (32,), 'depth': 8}
        with self.assertRaises(TypeError):
            cradle(0, data_type='image', task_type='magic', **options).init_tf_graph()

    def test_architecture_crossing_text(self):
        options = {'classes': 10, 'shape': (32,), 'depth': 8}
        tmp = cradle(0, data_type='image', **options)
        tmp2 = cradle(0, data_type='image', **options)
        pairing_type = [
            'father_architecture',
            'father_training',
            'father_architecture_layers',
            'father_architecture_slice_mother',
            'father_architecture_parameter']
        for t in pairing_type:
            self.assertIsInstance(
                peform_pairing(self.individ, tmp, tmp2, t), IndividBase)
