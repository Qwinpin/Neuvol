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


def peform_pairing(individ, father, mother, pairing_type):
    """
    Perform specific pairing according data type and pairing type
    """
    if individ.data_type == 'text':
        # start slice and end slice
        limitations = [2, 1]

    else:
        limitations = [1, 1]

    if pairing_type == 'father_architecture':
        return father_architecture_pairing(individ, father, mother)

    elif pairing_type == 'father_architecture_layers':
        return father_architecture_layers_pairing(individ, father, mother, limitations)

    elif pairing_type == 'father_architecture_parameter':
        return father_architecture_parameter_pairing(individ, father, mother, limitations)

    elif pairing_type == 'father_training':
        return father_training_pairing(individ, father, mother)

    elif pairing_type == 'father_data_processing':
        return father_data_processing_pairing(individ, father, mother)
    
    elif pairing_type == 'father_architecture_slice_mother':
        return father_architecture_slice_mother(individ, father, mother, limitations)

    else:
        return None


def father_architecture_pairing(individ, father, mother):
    """
    Father's architecture and mother's training and data
    """
    individ.architecture = father.architecture
    individ.training_parameters = mother.training_parameters
    individ.data_processing = mother.data_processing

    # change data processing parameter to avoid incompatibility
    individ.data_processing = father.data_processing

    return individ


def father_architecture_layers_pairing(individ, father, mother, limitations):
    """
    Select father's architecture and replace random block with mother's layer
    """
    changes_block = np.random.choice([i for i in range(limitations[0], len(father.architecture) - limitations[1])])
    alter_block = np.random.choice([i for i in range(limitations[0], len(mother.architecture) - limitations[1])])

    individ.architecture = father.architecture
    individ.architecture[changes_block] = mother.architecture[alter_block]
    individ.training_parameters = father.training_parameters
    individ.data_processing = father.data_processing

    return individ


def father_architecture_parameter_pairing(individ, father, mother, limitations):
    """
    Select father's architecture and change block parameters with mother's block
    dont touch first, second and last elements - input, embedding and dense(3),
    too many dependencies with text model
    select shared block
    """
    tmp_father = [block.type for block in father.architecture[limitations[0]:-limitations[1]]]
    tmp_mother = [block.type for block in mother.architecture[limitations[0]:-limitations[1]]]

    intersections = set(tmp_father) & set(tmp_mother)

    if not intersections:
        individ.architecture = father.architecture
        individ.training_parameters = father.training_parameters
        individ.data_processing = father.data_processing
    
    else:
        intersected_block = np.random.choice(list(intersections))

        # add 1, because we did not take into account first one|two block
        changes_block = tmp_father.index(intersected_block) + limitations[0]
        alter_block = tmp_mother.index(intersected_block) + limitations[0]

        individ.architecture = father.architecture
        individ.architecture[changes_block] = mother.architecture[alter_block]
        individ.training_parameters = father.training_parameters
        individ.data_processing = father.data_processing

    return individ


def father_data_processing_pairing(individ, father, mother):
    """
    Select father's data processing and mother's architecture and training
    change mother's embedding to avoid mismatchs in dimensions
    """
    individ.architecture = mother.architecture
    individ.training_parameters = mother.training_parameters
    individ.data_processing = father.data_processing

    # change data processing parameter to avoid incompatibility
    # it is used only with text processing
    if individ.data_type == 'text':
        individ.architecture[0] = father.architecture[0]
        individ.architecture[1] = father.architecture[1]

    return individ


def father_training_pairing(individ, father, mother):
    """
    Father's training and mother's architecture and data
    """
    individ.architecture = mother.architecture
    individ.training_parameters = father.training_parameters
    individ.data_processing = mother.data_processing

    return individ


def father_architecture_slice_mother(individ, father, mother, limitations):
    """
    Slice begining of the father and the end of the mother architecture and merge
    """
    father_cut_point = np.random.choice([i for i in range(limitations[0], len(father.architecture) - limitations[1])])
    mother_cut_point = np.random.choice([i for i in range(limitations[0], len(mother.architecture) - limitations[1])])

    tmp_individ = father.architecture[:father_cut_point] + mother.architecture[mother_cut_point:]
    
    individ.architecture = tmp_individ
    individ.data_processing = father.data_processing
    individ.training_parameters = father.training_parameters

    return individ
