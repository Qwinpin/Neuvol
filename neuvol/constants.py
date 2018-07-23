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


FLOAT32 = np.float32

# Training parameters
TRAINING = {
    'batchs': [i for i in range(8, 512, 32)],
    'epochs': [i for i in range(1, 25) if i % 2],
    'optimizer': ['adam', 'RMSprop'],
    'optimizer_decay': [FLOAT32(i / 10000) for i in range(1, 500, 5)],
    'optimizer_lr': [FLOAT32(i / 10000) for i in range(1, 500, 5)]}

# Specific parameters
SPECIAL = {
    'embedding': {
        'vocabular': [30000],
        'sentences_length': [10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150],
        'embedding_dim': [64, 128, 200, 300],
        'trainable': [False, True]}}

LAYERS_POOL = {
    'bi': {
        'units': [1, 2, 4, 8, 12, 16],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 5)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2]},

    'lstm': {
        'units': [1, 2, 4, 8, 12, 16],
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 5)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2]},

    'cnn': {
        'filters': [4, 8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7],
        'strides': [1, 2, 3],
        'padding': ['valid', 'same', 'causal'],
        'activation': ['tanh', 'relu'],
        'dilation_rate': [1, 2, 3]},

    'dense': {
        'units': [16, 64, 128, 256, 512],
        'activation': ['softmax', 'sigmoid']},

    'dropout': {'rate': [FLOAT32(i / 100) for i in range(5, 95, 5)]}}

POOL_SIZE = len(LAYERS_POOL)
