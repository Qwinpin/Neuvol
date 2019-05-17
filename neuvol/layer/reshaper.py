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

from .layer import Layer


def calculate_shape(prev_layer, layer):
    """
    Calculate shape and rank of the output tensor
    """
    prev_shape = prev_layer.config['shape']
    prev_rank = prev_layer.config['rank']

    if layer.type == 'lstm' or layer.type == 'bi':
        if layer.config['return_sequences']:
            rank = 3
        else:
            rank = 2
    elif layer.type == 'embedding':
        rank = 3
    # only [lstm, ...] change the output rank of the tensor
    else:
        rank = prev_rank

    if layer.type == 'cnn' or layer.type == 'cnn2':
        filters = layer.config['filters']
        kernel_size = layer.config['kernel_size']
        if kernel_size % 2 == 0:
            align = 1
        else:
            align = 0
        padding = layer.config['padding']
        strides = layer.config['strides']
        dilation_rate = layer.config['dilation_rate']

        if padding == 'valid':
            if dilation_rate != 1:
                out = [(i - kernel_size - (kernel_size - 1) * (dilation_rate - 1)) // strides + 1 - align
                       for i in prev_shape[1:-1]]
            else:
                out = [((i - kernel_size) // strides + 1 - align) for i in prev_shape[1:-1]]

        elif padding == 'same':
            out = [((i - kernel_size + (2 * (kernel_size // 2))) // strides + 1 - align) for i in prev_shape[1:-1]]

        elif padding == 'causal':
            out = [(i - kernel_size - (kernel_size - 1) * (dilation_rate - 1)) // strides + 1 - align
                   for i in prev_shape[1:-1]]

        for i in out:
            # if some of the layer too small - change the padding
            if i <= 0:
                layer.config['padding'] = 'same'
                rank, shape = calculate_shape(prev_layer, layer)
                return rank, shape

        shape = (None, *out, filters)

    elif layer.type == 'max_pool' or layer.type == 'max_pool2':
        kernel_size = layer.config['pool_size']
        strides = layer.config['strides']
        padding = layer.config['padding']

        if padding == 'same':
            out = [((i + 2*(kernel_size // 2) - kernel_size) // strides + 1) for i in prev_shape[1:-1]]
        else:
            out = [((i - kernel_size) // strides + 1) for i in prev_shape[1:-1]]

        shape = (None, *out, prev_shape[-1])

    elif layer.type == 'lstm' or layer.type == 'bi':
        if layer.type == 'bi':
            multiplier = 2
        else:
            multiplier = 1

        if layer.config['return_sequences']:
            shape = (None, prev_shape[1:-1], multiplier*layer.config['units'])
        else:
            shape = (None, multiplier*layer.config['units'])

    elif layer.type == 'dense':
        shape = (*prev_shape[:-1], layer.config['units'])

    elif layer.type == 'embedding':
        shape = (None, layer.config['sentences_length'], layer.config['embedding_dim'])

    elif layer.type == 'flatten':
        shape = (None, np.prod(prev_shape[1:]))

    else:
        shape = prev_shape

    return rank, shape


def reshaper_shape(difference, prev_layer):
    if difference > 0:
        # new shape is a dimensions flatten together, if we need to move from rank 5 to 2
        # we should take all dim from 1 (first - batch size, always None) and till the
        # difference + 1
        # (None, 12, 124, 124, 10) -> rank 2 -> (None, 12*124*124*10)
        # (None, 12, 124, 124, 10) -> rank 4 -> (None, 12, 124*124, 10)
        print(prev_layer.config['shape'], difference)
        new_shape = (
            None,
            *prev_layer.config['shape'][1:-(difference + 2)],
            np.prod(prev_layer.config['shape'][-(difference + 1): -1]),
            prev_layer.config['shape'][-1])

    elif difference < 0:
        # simple reshape with 1 dims
        # add new dims to remove the difference
        new_shape = (
            None,
            *prev_layer.config['shape'][1:],
            *([1] * abs(difference)))

    else:
        new_shape = prev_layer.config['shape']

    return new_shape


def reshaper(prev_layer, layer):
    """
    Restore compability between layer with diff ranks
    """
    if layer.config['input_rank'] is None:
        difference = 0
    else:
        difference = (prev_layer.config['rank'] - layer.config['input_rank'])

    if difference == 0:
        return None

    new_shape = reshaper_shape(difference, prev_layer)

    modifier = Layer('reshape')
    modifier.config['target_shape'] = new_shape[1:]
    modifier.config['shape'] = new_shape
    modifier.config['input_rank'] = prev_layer.config['rank']
    modifier.config['rank'] = layer.config['input_rank']

    return modifier


def merger_mass(layers):
    shape_modifiers = []
    for layer in layers:
        new_shape = (None, np.prod(layer.config['shape'][1:]))

        reshape = Layer('reshape')
        reshape.config['target_shape'] = new_shape[1:]
        reshape.config['shape'] = new_shape
        reshape.config['input_rank'] = layer.config['rank']
        reshape.config['rank'] = 2

        shape_modifiers.append(reshape)

    modifier = Layer('concat')

    shapes = [shape_modifiers[i].config['shape'][1:] for i, layer in enumerate(layers)]
    modifier.config['shape'] = (None, np.sum(shapes))
    modifier.config['rank'] = 2

    return modifier, shape_modifiers
