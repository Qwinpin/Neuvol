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
    'layers_number': [i for i in range(1, 10)]
}

# Training parameters
TRAINING = {
    'batchs': [i for i in range(8, 512, 32)],
    'epochs': [i for i in range(1, 100) if i % 2],
    'optimizer': ['adam', 'RMSprop'],
    'optimizer_decay': [FLOAT32(i / 10000) for i in range(1, 500, 1)],
    'optimizer_lr': [FLOAT32(i / 10000) for i in range(1, 500, 1)]}

# Specific parameters
SPECIAL = {
    'embedding': {
        'vocabular': [30000],
        'sentences_length': [i for i in range(1, 150, 1)],
        'embedding_dim': [i for i in range(32, 300, 2)],
        'trainable': [False, True]}}

LAYERS_POOL = {
    'bi': {
        'units': [i for i in range(1, 32, 1)],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 1)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2],
        'return_sequences': [True, False]},

    'lstm': {
        'units': [i for i in range(1, 32, 1)],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 1)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2],
        'return_sequences': [True, False]},

    'cnn': {
        'filters': [i for i in range(4, 128, 2)],
        'kernel_size': [i for i in range(1, 9, 1)],
        'strides': [1, 2, 3],
        'padding': ['valid', 'same', 'causal'],
        'activation': ['tanh', 'relu'],
        'dilation_rate': [1, 2, 3]},

    'cnn2': {
        'filters': [i for i in range(4, 128, 2)],
        'kernel_size': [i for i in range(1, 9, 1)],
        'strides': [1, 2, 3],
        'padding': ['valid', 'same'],
        'activation': ['tanh', 'relu'],
        'dilation_rate': [1, 2, 3]},

    'max_pool': {
        'pool_size': [i for i in range(2, 16)],
        'strides': [i for i in range(2, 8)],
        'padding': ['valid', 'same']},

    'max_pool2': {
        'pool_size': [i for i in range(2, 16)],
        'strides': [i for i in range(2, 8)],
        'padding': ['valid', 'same']},

    'dense': {
        'units': [i for i in range(4, 512, 2)],
        'activation': ['softmax', 'sigmoid']},

    'dropout': {'rate': [FLOAT32(i / 100) for i in range(5, 95, 1)]}}

POOL_SIZE = len(LAYERS_POOL)
