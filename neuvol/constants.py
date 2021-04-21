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
from collections import namedtuple

import faker
import numpy as np


EVENT = namedtuple('event', ['type', 'stage'])
FAKE = faker.Faker()
FLOAT32 = np.float32

# General parameters
GENERAL = {
    'layers_number': [i for i in range(1, 10)],
    'mutation_type': ['add_layer', 'add_connection', 'remove_connection', 'remove_layer'],
    'mutation_rate_splitting': 1.5,
    'mutation_rate_merge': 1.5,
    'mutation_splitting': {
        'rates': [0.7, 0.2, 0.07, 0.03],
        'number_of_splits': [2, 3, 4, 5]
    },
    'graph_parser': {
        'depth': 5,
        'min_size': 1
    }
}

# Training parameters
TRAINING = {
    'batchs': [4],  # [i for i in range(8, 512, 32)],
    'epochs': [200],  # [i for i in range(1, 100) if i % 2],
    'optimizer': ['adam'],  # ['adam', 'RMSprop'],
    'optimizer_decay': [FLOAT32(i / 10000) for i in range(1, 500, 1)],
    'optimizer_lr': [FLOAT32(i / 10000) for i in range(1, 500, 1)]}

# Specific parameters
SPECIAL = {
    'embedding': {
        'input_rank': [2],
        'vocabular': [30000],
        'sentences_length': [i for i in range(1, 150, 1)],
        'embedding_dim': [i for i in range(32, 300, 2)],
        'trainable': [False, True]},
    'zeropadding1D': {
        'input_rank': [],  # set manually
        'padding': []},
    'reshape': {  # set manually
        'input_rank': [],
        'target_shape': []},
    'flatten': {
        'input_rank': [], },
    'concat': {
        'input_rank': [],
    }}

LAYERS_POOL = {
    'lstm': {
        'input_rank': [3],
        'units': [i for i in range(1, 32, 1)],
        'hidden_size': [32, 64, 128, 256, 512, 768, 1024],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 1)],
        'activation': ['tanh', 'relu', None],
        'implementation': [1, 2],
        'bidirectional': [False, True],
        'return_sequences': [True, False]},

    'cnn': {
        'input_rank': [3],
        'filters': [i for i in range(0, 128, 16)][1:] + [1],
        'kernel_size': [i for i in range(1, 32, 2)],
        # 'strides': [1],
        'strides': [1, 2, 3],
        'padding_mode': ['valid'],
        'activation': ['tanh', 'relu', None],
        'dilation_rate': [1, 2, 3]},

    'cnn2': {
        'input_rank': [4],
        'filters': [i for i in range(0, 128, 16)][1:] + [1],
        'kernel_size': [i for i in range(1, 32, 2)],
        # 'strides': [1],
        'strides': [1, 2, 3],
        'padding_mode': ['valid'],
        # 'padding': ['same'],
        'activation': ['tanh', 'relu', None],
        'dilation_rate': [1, 2, 3]},

    'max_pool': {
        'input_rank': [3],
        'pool_size': [i for i in range(0, 16, 2)][1:],
        'strides': [i for i in range(2, 8)],
        'dilation_rate': [1, 2, 3],
        'padding_mode': [None],
        # 'padding': ['same'],
        },

    'max_pool2': {
        'input_rank': [4],
        'pool_size': [i for i in range(0, 16, 2)][1:],
        'dilation_rate': [1, 2, 3],
        'strides': [i for i in range(2, 8)],
        'padding_mode': [None]},

    'dense': {
        'input_rank': [],
        'units': [i for i in range(0, 512, 16)][1:] + [1],
        'activation': ['relu', 'softmax', 'sigmoid', None]},

    'dropout': {
        'input_rank': [],
        'rate': [FLOAT32(i / 100) for i in range(0, 50, 5)]},

    'decnn2': {
        'input_rank': [4],
        'filters': [i for i in range(0, 128, 16)][1:] + [1],
        'kernel_size': [i for i in range(1, 11, 2)],
        'strides': [1],
        # 'strides': [1, 2, 3],
        'padding_mode': ['valid'],
        'output_padding': [i for i in range(0, 5)],
        'activation': ['tanh', 'relu', None],
        'dilation_rate': [1, 2, 3]}, }


POOL_SIZE = len(LAYERS_POOL)
