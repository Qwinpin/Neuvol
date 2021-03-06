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
import copy
from keras.layers import (Bidirectional, concatenate, Conv1D, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPool1D, MaxPool2D, Reshape, RepeatVector,
                          SeparableConv1D, SeparableConv2D, Conv2DTranspose)
from keras.layers.recurrent import LSTM
import math
import numpy as np

from ..constants import LAYERS_POOL, SPECIAL
from ..utils import dump

# TODO: layers serialisation
# TODO: shape calculation recursion error while net initializing
def Layer(layer_type, distribution, options=None, previous_layer=None, next_layer=None, data_load=None):
    """
    Factory for the Layers instances
    """
    if layer_type in LAYERS_MAP:
        layer = LAYERS_MAP[layer_type](layer_type=layer_type, distribution=distribution, previous_layer=previous_layer, next_layer=next_layer, options=options, data_load=data_load)
    elif layer_type in distribution.CUSTOM_LAYERS_MAP.keys():
        layer = copy.deepcopy(distribution.CUSTOM_LAYERS_MAP[layer_type])
    else:
        raise TypeError()

    # if data_load is not None:
    #     layer.config = data_load['config']

    return layer


class LayerBase:
    """
    Single layer class with compatibility checking
    In order to keep logical connectivity with Keras layer instances
    __call__ method is implemented.
    When calling ranks comparison is performed - in case of incompatibilities
    of rank add reshape layer.
    """

    def __init__(self, layer_type, distribution, previous_layer=None, next_layer=None, options=None, data_load=None):
        self.config = {}
        self.layer_type = layer_type
        self.distribution = distribution
        self.options = options
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        if data_load is not None:
            self.load(data_load)
        elif layer_type is not None:
            self._init_parameters()
            self._check_compatibility()

    def __call__(self, net, previous_layer):
        """
        Add layer to a network tail, previous layer is required for shape and rank check
        In case of multiple layers concatenation layer is injected
        """
        # in case of concatenation
        if isinstance(net, list):
            concat_layer = Layer('concat', self.distribution)
            net = concat_layer(net, previous_layer)

            previous_layer = concat_layer

        reshape_layer = self._init_reshape_layer(previous_layer)

        if reshape_layer is None:
            self.config['rank'] = self.calculate_rank(previous_layer)
            self.config['shape'] = self.calculate_shape(previous_layer)
        else:
            self.config['rank'] = self.calculate_rank(reshape_layer)
            self.config['shape'] = self.calculate_shape(reshape_layer)

        if self.config['shape'] is None:
            self.config['state'] = 'broken'
            return net

        if reshape_layer is not None:
            new_new = reshape_layer(net, previous_layer)
        else:
            new_new = net

        layer_instance = self.init_layer(previous_layer)
        try:
            new_net = layer_instance(new_new)
        except:
            raise

        return new_net

    def init_layer(self, previous_layer):
        """
        Each layer type has its own initialization
        """
        return ...

    def _init_reshape_layer(self, previous_layer):
        """
        Add reshape layer if ranks is different
        """
        if self.config['input_rank'] != previous_layer.rank:
            reshape_layer = reshaper(previous_layer, self, self.distribution)
        else:
            reshape_layer = None

        return reshape_layer

    def _init_parameters(self):
        """
        Get random values of all required parameters
        """
        variables = list(LAYERS_POOL[self.layer_type])
        for parameter in variables:
            self.config[parameter] = self.distribution.layer_parameters(self.layer_type, parameter)

        if self.options is not None and self.options.get('input_rank') is not None:
            self.config['input_rank'] = self.options['input_rank']

    def _check_compatibility(self):
        pass

    def calculate_shape(self, previous_layer):
        """
        Shape calculator for the output

        Arguments:
            previous_layer {Layer} -- previous layer, which is connected to the current
        """
        previous_shape = previous_layer.shape
        new_shape = previous_shape

        return new_shape

    def calculate_rank(self, previous_layer):
        previous_rank = previous_layer.rank
        new_rank = previous_rank

        return new_rank

    @property
    def shape(self):
        return self.config['shape']

    @property
    def rank(self):
        return self.config['rank']

    @shape.setter
    def shape(self, value):
        self.config['shape'] = value

    @rank.setter
    def rank(self, value):
        self.config['shape'] = value

    def dump(self):
        buffer = {}
        buffer['config'] = self.config
        buffer['options'] = self.options
        buffer['layer_type'] = self.layer_type

        return buffer

    def load(self, data_load):
        self.config = data_load['config']
        self.options = data_load['options']
        self.layer_type = data_load['layer_type']


class LayerSpecialBase(LayerBase):
    def _init_parameters(self):
        variables = list(SPECIAL[self.layer_type])
        for parameter in variables:
            self.config[parameter] = self.distribution.layer_parameters(self.layer_type, parameter)


class LayerComplex(LayerBase):
    # TODO: saver/loader
    def __init__(self, layer_matrix, initiated_layers, width=None):
        self.matrix = layer_matrix
        self.layers_index_reverse = initiated_layers
        self.size = len(layer_matrix)
        self.width = width or len(initiated_layers)

        self.config = {}

    def __call__(self, net, previous_layer):
        tail_map = {}

        for column in range(len(self.layers_index_reverse)):
            last_layer = self.rec_imposer_sub_graph(column, tail_map, net, previous_layer)

        self.config = {}
        self.config['rank'] = self.layers_index_reverse[last_layer].rank
        self.config['shape'] = self.layers_index_reverse[last_layer].shape

        return tail_map[last_layer]

    def rec_imposer_sub_graph(self, column, tails_map, net_tail=None, net_tail_layer=None):
        if tails_map.get(column, None) is not None:
            return None

        column_values = self.matrix[:, column]
        connections = np.where(column_values == 1)[0]

        for index in connections:
            if tails_map.get(index, None) is None:
                self.rec_imposer_sub_graph(self, index, tails_map)

        if not connections.size > 0:
            if column == 0:
                tails_map[column] = self.layers_index_reverse[column](net_tail, net_tail_layer)
            last_non_zero = column

        elif connections.size > 1:
            tails_to_call = [tails_map[i] for i in connections]
            layers_to_call = [self.layers_index_reverse[i] for i in connections]

            tails_map[column] = self.layers_index_reverse[column](tails_to_call, layers_to_call)
            last_non_zero = column

        else:
            tails_map[column] = self.layers_index_reverse[column](tails_map[connections[0]], self.layers_index_reverse[connections[0]])
            last_non_zero = column

        return last_non_zero

    def dump_complex(self):
        buffer = {}
        buffer['config'] = self.config
        buffer['matrix'] = self.matrix.tolist()
        buffer['layers'] = {}
        for i, _layer in self.layers_index_reverse.items():
            buffer['layers'][i] = _layer.dump()

        buffer['size'] = self.size
        buffer['width'] = self.width

        return buffer

class LayerLSTM(LayerBase):
    def _check_compatibility(self):
        if self.next_layer is not None and self.next_layer != 'last_dense':
            self.config['return_sequences'] = True
        else:
            self.config['return_sequences'] = False

    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = LSTM(
            units=self.config['units'],
            recurrent_dropout=self.config['recurrent_dropout'],
            activation=self.config['activation'],
            implementation=self.config['implementation'],
            return_sequences=self.config['return_sequences'])

        return layer_tf

    def calculate_rank(self, previous_layer):
        if self.config['return_sequences']:
            rank = 3
        else:
            rank = 2

        return rank

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape

        if self.config['return_sequences']:
            shape = (None, previous_shape[1:-1], self.config['units'])
        else:
            shape = (None, self.config['units'])

        return shape


class LayerBiLSTM(LayerLSTM):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Bidirectional(
            LSTM(
                units=self.config['units'],
                recurrent_dropout=self.config['recurrent_dropout'],
                activation=self.config['activation'],
                implementation=self.config['implementation'],
                return_sequences=self.config['return_sequences']))

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape

        if self.config['return_sequences']:
            shape = (None, previous_shape[1:-1], 2*self.config['units'])
        else:
            shape = (None, 2*self.config['units'])

        return shape


class LayerCNN1D(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Conv1D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size']],
            strides=[self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf

    def _check_compatibility(self):
        super()._check_compatibility()
        if self.config['dilation_rate'] > 1:
            self.config['strides'] = 1

        elif self.config['dilation_rate'] == 1 and self.config['padding'] == 'causal':
            self.config['padding'] = 'same'

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        filters = self.config['filters']
        kernel_size = self.config['kernel_size']
        # keep this hack, need to valid
        if kernel_size % 2 == 0:
            align = 0
        else:
            align = 0

        padding = self.config['padding']
        strides = self.config['strides']
        dilation_rate = self.config['dilation_rate']

        if padding == 'valid':
            if dilation_rate != 1:
                out = [(i - kernel_size - (kernel_size - 1) * (dilation_rate - 1)) // strides + 1 - align
                       for i in previous_shape[1:-1]]
            else:
                out = [((i - kernel_size) // strides + 1 - align) for i in previous_shape[1:-1]]

        elif padding == 'same':
            # out = [((i - kernel_size + (2 * (kernel_size // 2))) // strides + 1 - align) for i in previous_shape[1:-1]]
            out = [math.ceil(i / strides) for i in previous_shape[1:-1]]

        elif padding == 'causal':
            out = [i for i in previous_shape[1:-1]]
            # out = [(i - kernel_size - (kernel_size - 1) * (dilation_rate - 1)) // strides + 1 - align
            #        for i in previous_shape[1:-1]]

        for i in out:
            # if some of the layer too small - change the padding
            if i <= 0:
                self.config['padding'] = 'same'
                shape = self.calculate_shape(previous_layer)
                return shape

        shape = (None, *out, filters)

        return shape


class LayerCNN2D(LayerCNN1D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Conv2D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerMaxPool1D(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = MaxPool1D(
            pool_size=[self.config['pool_size']],
            strides=[self.config['strides']],
            padding=self.config['padding'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape

        kernel_size = self.config['pool_size']
        strides = self.config['strides']
        padding = self.config['padding']
        if kernel_size % 2 != 0:
            align = 0
        else:
            align = 0

        if padding == 'same':
            # out = [((i + 2*(kernel_size // 2) - kernel_size) // strides + 1 - align) for i in previous_shape[1:-1]]
            out = [math.ceil(i / strides) for i in previous_shape[1:-1]]
        else:
            out = [((i - kernel_size) // strides + 1 - align) for i in previous_shape[1:-1]]

        for i in out:
            # if some of the layer too small - change the padding
            if i <= 0 and padding != 'same':
                self.config['padding'] = 'same'
                shape = self.calculate_shape(previous_layer)
                return shape
            elif i <= 0:
                return None

        shape = (None, *out, previous_shape[-1])

        return shape


class LayerMaxPool2D(LayerMaxPool1D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = MaxPool2D(
            pool_size=[self.config['pool_size'], self.config['pool_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'])

        return layer_tf


class LayerDense(LayerBase):
    def _init_parameters(self):
        if self.layer_type == 'last_dense':
            variables = list(LAYERS_POOL['dense'])
            for parameter in variables:
                self.config[parameter] = self.distribution.layer_parameters('dense', parameter)
        else:
            super()._init_parameters()

    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Dense(
            units=self.config['units'],
            activation=self.config['activation'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (*previous_shape[:-1], self.config['units'])

        return shape


class LayerInput(LayerBase):
    def _init_parameters(self):
        self.config['shape'] = self.options['shape']
        self.config['rank'] = len(self.options['shape'])

    def init_layer(self, previous_layer):
        layer_tf = Input(
            shape=self.config['shape'][1:])  # all except first None, related to batch size

        return layer_tf


class LayerEmbedding(LayerSpecialBase):
    def _init_parameters(self):
        super()._init_parameters()
        self.config['sentences_length'] = self.options['shape'][0]

    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Embedding(
            input_dim=self.config['vocabular'],
            output_dim=self.config['embedding_dim'],
            input_length=self.config['sentences_length'],
            trainable=self.config['trainable'])

        return layer_tf

    def calculate_rank(self, previous_layer):
        rank = 3

        return rank

    def calculate_shape(self, previous_layer):
        shape = (None, self.config['sentences_length'], self.config['embedding_dim'])

        return shape


class LayerFlatten(LayerSpecialBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Flatten()

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (None, np.prod(previous_shape[1:]))

        return shape


class LayerConcat(LayerSpecialBase):
    # TODO: smart merger according to most frequent shape size
    def __call__(self, nets, previous_layers):
        reshape_layers, axis = self.merger_mass(previous_layers)
        new_nets = []
        if axis == -1:
            for i, reshape_layer in enumerate(reshape_layers):
                new_nets.append(reshape_layer(nets[i], previous_layers[i]))
        else:
            new_nets = nets
        # new_nets = [reshape_layer(nets[i], previous_layers[i]) for i, reshape_layer in enumerate(reshape_layers)]

        new_net = concatenate(new_nets, axis)
        return new_net

    def merger_mass(self, layers):
        shape_modifiers = []
        shapes = [layer.shape for layer in layers]

        # if all shapes are equal in.. shape
        for i in range(1, max([len(shape) for shape in shapes])):
            # check equality
            key = True
            for shape in shapes[1:]:
                if len(shape) != len(shapes[0]) or len(shape) != max([len(shape) for shape in shapes]):
                    key = False
                    break
                tmp1, tmp2 = list(shape), list(shapes[0])
                tmp1.pop(i)
                tmp2.pop(i)

                if tmp1 != tmp2:
                    key = False
                    break
            axis = i

            if key:
                new_shape = np.array(shapes[0])
                new_shape[axis] = np.sum(np.array(shapes)[:, axis])
                self.config['shape'] = new_shape
                self.config['rank'] = len(new_shape)
                return layers, axis

        for layer in layers:
            new_shape = (None, np.prod(layer.config['shape'][1:]))

            reshape = Layer('reshape', self.distribution)
            reshape.config['target_shape'] = new_shape[1:]
            reshape.config['shape'] = new_shape
            reshape.config['input_rank'] = layer.config['rank']
            reshape.config['rank'] = 2

            shape_modifiers.append(reshape)

        shapes = [shape_modifiers[i].config['shape'][1:] for i, layer in enumerate(layers)]
        self.config['shape'] = (None, np.sum(shapes))
        self.config['rank'] = 2

        return shape_modifiers, -1


class LayerReshape(LayerSpecialBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Reshape(
            target_shape=self.config['target_shape'],)

        return layer_tf


class LayerDropout(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Dropout(rate=self.config['rate'])

        return layer_tf


class LayerRepeatVector(LayerBase):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = RepeatVector(n=self.config['n'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        shape = (None, self.config['n'], *previous_shape[1:])

        return shape

    def calculate_rank(self, previous_layer):
        previous_rank = previous_layer.rank
        rank = previous_rank + 1

        return rank


class LayerSepCNN1D(LayerCNN1D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = SeparableConv1D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size']],
            strides=[self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerSepCNN2D(LayerCNN2D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = SeparableConv2D(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf


class LayerDeCNN2D(LayerCNN2D):
    def init_layer(self, previous_layer):
        super().init_layer(previous_layer)
        layer_tf = Conv2DTranspose(
            filters=self.config['filters'],
            kernel_size=[self.config['kernel_size'], self.config['kernel_size']],
            strides=[self.config['strides'], self.config['strides']],
            padding=self.config['padding'],
            output_padding=self.config['output_padding'],
            dilation_rate=tuple([self.config['dilation_rate'], self.config['dilation_rate']]),
            activation=self.config['activation'])

        return layer_tf

    def calculate_shape(self, previous_layer):
        previous_shape = previous_layer.shape
        filters = self.config['filters']
        kernel_size = self.config['kernel_size']

        padding = self.config['padding']
        output_padding = self.config['output_padding']
        strides = self.config['strides']
        dilation_rate = self.config['dilation_rate']

        if padding == 'valid':
            if output_padding is None:
                output_padding = 0
            out = [(i - 1) * strides + kernel_size + (kernel_size - 1) * (dilation_rate - 1) + output_padding for i in previous_shape[1:-1]]

        elif padding == 'same':
            if output_padding is None:
                out = [i * strides for i in previous_shape[1:-1]]
            else:
                out = [(i - 1) * strides + kernel_size - 2 * (kernel_size // 2) + output_padding for i in previous_shape[1:-1]]

        shape = (None, *out, filters)

        return shape


LAYERS_MAP = {
    'input': LayerInput,
    'lstm': LayerLSTM,
    'bi': LayerBiLSTM,
    'cnn': LayerCNN1D,
    'cnn2': LayerCNN2D,
    'max_pool': LayerMaxPool1D,
    'max_pool2': LayerMaxPool2D,
    'dense': LayerDense,
    'embedding': LayerEmbedding,
    'flatten': LayerFlatten,
    'concat': LayerConcat,
    'reshape': LayerReshape,
    'dropout': LayerDropout,
    'repeatvector': LayerRepeatVector,
    'separablecnn': LayerSepCNN1D,
    'separablecnn2': LayerSepCNN2D,
    'decnn2': LayerDeCNN2D
}


def reshaper_shape(difference, prev_layer):
    if difference > 0:
        # new shape is a dimensions flatten together, if we need to move from rank 5 to 2
        # we should take all dim from 1 (first - batch size, always None) and till the
        # difference + 1
        # (None, 12, 124, 124, 10) -> rank 2 -> (None, 12*124*124*10)
        # (None, 12, 124, 124, 10) -> rank 4 -> (None, 12, 124*124, 10)
        prev_rank = prev_layer.rank

        # how many elements will be multiplied
        window = difference + 1
        # how many elements will not be multiplied
        # -1 used because None elements is not taken into account
        free_elements = prev_rank - window - 1

        if free_elements == 0:
            tail_ind = None
            front_ind = 1

        else:
            tail_ind = -1
            front_ind = free_elements

        front = prev_layer.config['shape'][1: front_ind]
        mid = np.prod(prev_layer.config['shape'][front_ind: tail_ind])

        if tail_ind is None:
            tail = []
        else:
            tail = prev_layer.config['shape'][-1:]

        new_shape = (
            None,
            *front,
            mid,
            *tail)

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


def reshaper(prev_layer, layer, distribution):
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

    modifier = Layer('reshape', distribution)
    modifier.config['target_shape'] = new_shape[1:]
    modifier.config['shape'] = new_shape
    modifier.config['input_rank'] = prev_layer.config['rank']
    modifier.config['rank'] = layer.config['input_rank']

    return modifier
