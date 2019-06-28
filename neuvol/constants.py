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
    'mutation_type': ['add_layer', 'add_connection', 'remove_connection']
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
    'bi': {
        'input_rank': [3],
        'units': [i for i in range(1, 32, 1)],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 1)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2],
        'return_sequences': [True, False]},

    'lstm': {
        'input_rank': [3],
        'units': [i for i in range(1, 32, 1)],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 1)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2],
        'return_sequences': [True, False]},

    'cnn': {
        'input_rank': [3],
        'filters': [i for i in range(4, 128, 2)],
        'kernel_size': [i for i in range(1, 11, 2)],
        'strides': [1, 2, 3],
        'padding': ['valid', 'same', 'causal'],
        'activation': ['tanh', 'relu'],
        'dilation_rate': [1, 2, 3]},

    'cnn2': {
        'input_rank': [4],
        'filters': [i for i in range(1, 128, 1)],
        'kernel_size': [i for i in range(1, 11, 2)],
        'strides': [1, 2, 3],
        'padding': ['valid', 'same'],
        'activation': ['tanh', 'relu'],
        'dilation_rate': [1, 2, 3]},

    'max_pool': {
        'input_rank': [3],
        'pool_size': [i for i in range(1, 16, 2)],
        'strides': [i for i in range(2, 8)],
        'padding': ['valid', 'same']},

    'max_pool2': {
        'input_rank': [4],
        'pool_size': [i for i in range(1, 16, 2)],
        'strides': [i for i in range(2, 8)],
        'padding': ['valid', 'same']},

    'dense': {
        'input_rank': [],
        'units': [i for i in range(4, 512, 2)],
        'activation': ['softmax', 'sigmoid']},

    'dropout': {
        'input_rank': [],
        'rate': [FLOAT32(i / 100) for i in range(5, 95, 1)]}}

POOL_SIZE = len(LAYERS_POOL)
