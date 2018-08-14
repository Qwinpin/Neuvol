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

from .base_pairing import CrosserBase
from .pairing_modules import peform_pairing
from ..architecture.individ_image import IndividImage


class CrosserImage(CrosserBase):
    """
    Crossing class for visual data
    """
    @staticmethod
    def cross(father, mother, stage):
        """
        New individ parameters according its parents (only 2 now, classic)
        """
        # father_architecture - chose architecture from first individ and text
        # and train from second
        # father_training - only training config from first one
        # father_arch_layers - select overlapping layers
        # and replace parameters from the first architecture
        # with parameters from the second

        individ = IndividImage(stage, parents=[father, mother], **father.options)
        pairing_type = np.random.choice([
            'father_architecture',
            'father_training',
            'father_architecture_layers',
            'father_architecture_slice_mmother',
            'father_architecture_parameter'])

        return peform_pairing(individ, father, mother, pairing_type)
