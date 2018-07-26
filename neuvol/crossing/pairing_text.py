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

from .pairing_base import CrosserBase
from ..architecture.individ_text import IndividText


class CrosserText(CrosserBase):
    """
    Crossing class for textual data
    """
    def __init__(self):
        pass

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

        individ = IndividText(stage, parents=[father, mother], **father.options)
        pairing_type = np.random.choice([
            'father_architecture',
            'father_training',
            'father_architecture_layers',
            'father_architecture_parameter',
            'father_data_processing'])

        if pairing_type == 'father_architecture':
            # Father's architecture and mother's training and data
            individ.architecture = father.architecture
            individ.training_parameters = mother.training_parameters
            individ.data_processing = mother.data_processing

            # change data processing parameter to avoid incompatibility
            individ.data_processing['sentences_length'] = father.data_processing['sentences_length']

        elif pairing_type == 'father_training':
            # Father's training and mother's architecture and data
            individ.architecture = mother.architecture
            individ.training_parameters = father.training_parameters
            individ.data_processing = mother.data_processing

        elif pairing_type == 'father_architecture_layers':
            # Select father's architecture and replace random layer with mother's layer
            changes_layer = np.random.choice([i for i in range(1, len(father.architecture))])
            alter_layer = np.random.choice([i for i in range(1, len(mother.architecture))])

            individ.architecture = father.architecture
            individ.architecture[changes_layer] = mother.architecture[alter_layer]
            individ.training_parameters = father.training_parameters
            individ.data_processing = father.data_processing

        elif pairing_type == 'father_architecture_parameter':
            # Select father's architecture and change layer parameters with mother's layer
            # dont touch first and last elements - embedding and dense(3),
            # too many dependencies with text model
            # select common layer
            tmp_father = [layer.type for layer in father.architecture[1:-1]]
            tmp_mother = [layer.type for layer in mother.architecture[1:-1]]

            intersections = set(tmp_father) & set(tmp_mother)

            if not intersections:
                individ.architecture = father.architecture
                individ.training_parameters = father.training_parameters
                individ.data_processing = father.data_processing

            intersected_layer = np.random.choice(list(intersections))

            # add 1, because we did not take into account first layer
            changes_layer = tmp_father.index(intersected_layer) + 1
            alter_layer = tmp_mother.index(intersected_layer) + 1

            individ.architecture = father.architecture
            individ.architecture[changes_layer] = mother.architecture[alter_layer]
            individ.training_parameters = father.training_parameters
            individ.data_processing = father.data_processing

        elif pairing_type == 'father_data_processing':
            # Select father's data processing and mother's architecture and training
            # change mother's embedding to avoid mismatchs in dimensions
            individ.architecture = mother.architecture
            individ.training_parameters = mother.training_parameters
            individ.data_processing = father.data_processing

            # change data processing parameter to avoid incompatibility
            individ.architecture[0] = father.architecture[0]

        return individ
