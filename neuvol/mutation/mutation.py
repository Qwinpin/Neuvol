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

from ..constants import EVENT, TRAINING
from ..layer import Layer, LAYERS_POOL


class Mutator():
    def __init__(self):
        pass

    def mutate(self, network, stage):
        """
        Darwin was right. Change some part of individ with some probability
        """
        mutation_type = np.random.choice(['architecture', 'train', 'all'], p=(0.45, 0.45, 0.1))
        network.history = EVENT('Mutation', stage)
        architecture = network.architecture
        training_parameters = network.training_parameters
        data_processing = network.data_processing

        if mutation_type == 'architecture':
            # all - change the whole net
            mutation_size = np.random.choice(['all', 'part', 'parameters'], p=(0.3, 0.3, 0.4))

            if mutation_size == 'all':
                # If some parts of the architecture change, the processing of data must also change
                architecture = network.random_init_architecture()

            elif mutation_size == 'part':
                # select layer except the first and the last one - embedding and dense(*)
                mutation_layer = np.random.choice([i for i in range(1, len(architecture) - 1)])

                # find next layer to avoid incopabilities in neural architecture
                next_layer = network.architecture[mutation_layer + 1]
                new_layer = np.random.choice(list(LAYERS_POOL.keys()))
                layer = Layer(new_layer, next_layer=next_layer)

                architecture[mutation_layer] = layer

            elif mutation_size == 'parameters':
                # select layer except the first and the last one - embedding and dense(3)
                mutation_layer = np.random.choice([i for i in range(1, len(architecture) - 1)])

                # find next layer to avoid incopabilities in neural architecture
                next_layer = network.architecture[mutation_layer + 1]
                new_layer = network.architecture[mutation_layer].type

                architecture[mutation_layer] = Layer(new_layer, next_layer=next_layer)

        elif mutation_type == 'train':
            mutation_size = np.random.choice(['all', 'part'], p=(0.3, 0.7))

            if mutation_size == 'all':
                training_parameters = network.random_init_training()

            elif mutation_size == 'part':
                mutation_parameter = np.random.choice(list(TRAINING))
                new_training = network.random_init_training()
                training_parameters[mutation_parameter] = new_training[mutation_parameter]

        elif mutation_type == 'all':
            # change the whole individ - similar to death and rebirth
            architecture = network.random_init_architecture
            training_parameters = network._random_init_training

        data_processing['sentences_length'] = architecture[0].config['sentences_length']

        network.architecture = architecture
        network.data_processing = data_processing
        network.training_parameters = training_parameters

        return network
