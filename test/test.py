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
from architecture import Individ


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Start evolution test
        """
        print("==========")

    @classmethod
    def tearDownClass(cls):
        """
        Test end
        """
        print("==========")

    def setUp(self):
        print("Set up for [" + self.shortDescription() + "]")

    def tearDown(self):
        pass

    def test_init(self):
        """
        Initialization test
        """
        options = {'classes': 2}
        self.assertIsNotNone(0, Individ(stage=1, data_type='text', task_type='classification', parents=None, **options))

    # def test_err_init(self):
    #     """
    #     Wrong cases test
    #     without options
    #     """
    #     Individ(stage=1, data_type='text', task_type='classification', parents=None)

    def test_compile(self):
        options = {'classes': 2}
        a = Individ(stage=1, data_type='text', task_type='classification', parents=None, **options)
        self.assertEqual(len(a.init_tf_graph()), 3)
