import torch
import numpy as np


class Network(torch.nn.Module):
    def __init__(self, structure):
        super(Network, self).__init__()
        self.structure = structure
        self.layers_pool_inited = self.init_layers(self.structure)
        
    def init_layers(self, structure):
        # pool of layers, which should be initialised and connected
        layers_pool = [0]

        # pool of initialised layers
        layers_pool_inited = {}

        # pool of broken (invalid) layers) such as inconsistent number of dimensions
        layers_pool_removed = []

        while layers_pool:
            # take first layer in a pool
            layer_index = layers_pool[0]

            # find all connections before this layer
            enter_layers = set(np.where(self.structure.matrix[:, layer_index] == 1)[0])

            # check if some of previous layers were not initialized
            # that means - we should initialise them first
            not_inited_layers = [i for i in enter_layers if i not in (layers_pool_inited.keys())]
            not_inited_layers_selected = [layer for layer in not_inited_layers if layer not in layers_pool_removed]

            if not_inited_layers_selected:
                # remove layers, which are in pool already
                # this is possible due to complex connections with different orders
                not_inited_layers_selected = [layer for layer in not_inited_layers_selected if layer not in layers_pool]

                # add not initialised layers to the pool
                layers_pool.extend(not_inited_layers_selected)

                # current layer should be shift to the end of the queue
                acc = layers_pool.pop(0)
                layers_pool.append(acc)
                continue

            # take Layer instance of the previous layers
            input_layers = [self.structure.layers_index_reverse[layer] for layer in enter_layers]

            # layer without rank is broken and we ignore that
            input_layers = [layer for layer in input_layers if layer.config.get('rank', False)]
            enter_layers = [i for i in enter_layers if i not in layers_pool_removed]

            # if curent layer is the Input - initialise without any input connections
            if not input_layers and self.structure.layers_index_reverse[layer_index].layer_type == 'input':
                inited_layer = (None, None, self.structure.layers_index_reverse[layer_index].init_layer(None))

            # detect hanging node - some of mutations could remove connection to the layer
            elif not input_layers:
                layers_pool_removed.append(layers_pool.pop(0))
                continue

            # if there are multiple input connections
            elif len(input_layers) > 1:

                # this case does not require additional processing - all logic is inside Layer instance,
                # which handles multiple connections
                inited_layer = self.structure.layers_index_reverse[layer_index]([None for _ in range(len(input_layers))], input_layers)
            else:
                input_layers_inited = [layers_pool_inited[layer] for layer in enter_layers][0]
                inited_layer = self.structure.layers_index_reverse[layer_index](None, input_layers[0])

            # add new initialised layer
            layers_pool_inited[layer_index] = inited_layer
            setattr(self, 'layer_{}'.format(layer_index), inited_layer[2])

            # find outgoing connections and add them to the pool
            output_layers = [layer for layer in np.where(self.structure.matrix[layer_index] == 1)[0]
                                if layer not in layers_pool and layer not in layers_pool_inited.keys()]

            layers_pool.extend(output_layers)

            # remove current layer from the pool
            layers_pool.pop(layers_pool.index(layer_index))
            
        self.layers_pool_removed = layers_pool_removed
        return layers_pool_inited
        
    def forward(self, x):
        # pool of layers, which should be initialised and connected
        layers_pool = [0]
        buffer_x = {-1: x}
        last_value = None

        while layers_pool:
            # take first layer in a pool
            layer_index = layers_pool[0]
            # find all connections before this layer
            enter_layers = set(np.where(self.structure.matrix[:, layer_index] == 1)[0])
            enter_layers = [i for i in enter_layers if i not in self.layers_pool_removed]

            # check if some of previous layers were not initialized
            # that means - we should initialise them first
            not_inited_layers = [i for i in enter_layers if i not in (buffer_x.keys())]
            not_inited_layers_selected = [layer for layer in not_inited_layers if layer not in self.layers_pool_removed]

            if not_inited_layers_selected:
                # remove layers, which are in pool already
                # this is possible due to complex connections with different orders
                not_inited_layers_selected = [layer for layer in not_inited_layers_selected if layer not in layers_pool]

                # add not initialised layers to the pool
                layers_pool.extend(not_inited_layers_selected)

                # current layer should be shift to the end of the queue
                layers_pool.append(layers_pool.pop(0))
                continue

            # take Layer instance of the previous layers
            temp_x = [buffer_x[layer] for layer in enter_layers]

            # if curent layer is the Input - initialise without any input connections
            if not enter_layers and self.structure.layers_index_reverse[layer_index].layer_type == 'input':
                if self.layers_pool_inited[layer_index][0] is not None:
                    raise "Input layer is not the first one. Incorrect graph structure"

                if self.layers_pool_inited[layer_index][1] is not None:
                    reshaper = self.layers_pool_inited[layer_index][1]  # .init_layer(None)
                    temp_x = reshaper(buffer_x[-1])
                else:
                    temp_x = buffer_x[-1]

                result_x = self.process_layer_output(self.layers_pool_inited[layer_index][2](temp_x), self.structure.layers_index_reverse[layer_index].layer_type)
                buffer_x[layer_index] = result_x

            # detect hanging node - some of mutations could remove connection to the layer
            elif not enter_layers:
                continue

            # if there are multiple input connections
            elif len(enter_layers) > 1:
                if self.layers_pool_inited[layer_index][0] is not None:
                    reshapers = self.layers_pool_inited[layer_index][0][0]
                    axis = self.layers_pool_inited[layer_index][0][1]
                    if reshapers is not None:
                        reshapers = [i.init_layer(None) for i in reshapers]
                        temp_x = [r(temp_x[i]) for i, r in enumerate(reshapers)]
                    temp_x = torch.cat(temp_x, axis)

                if self.layers_pool_inited[layer_index][1] is not None:
                    temp_x = self.layers_pool_inited[layer_index][1](temp_x)

                result_x = self.process_layer_output(self.layers_pool_inited[layer_index][2](temp_x), self.structure.layers_index_reverse[layer_index].layer_type)
                buffer_x[layer_index] = result_x

            else:
                temp_x = temp_x[0]
                if self.layers_pool_inited[layer_index][1] is not None:
                    reshaper = self.layers_pool_inited[layer_index][1]  # .init_layer(None)
                    temp_x = reshaper(temp_x)

                result_x = self.process_layer_output(self.layers_pool_inited[layer_index][2](temp_x), self.structure.layers_index_reverse[layer_index].layer_type)
                buffer_x[layer_index] = result_x

            # find outgoing connections and add them to the pool
            output_layers = [layer for layer in np.where(self.structure.matrix[layer_index] == 1)[0]
                                if layer not in layers_pool and layer not in buffer_x.keys()]
            
            last_value = result_x
            layers_pool.extend(output_layers)

            # remove current layer from the pool
            layers_pool.pop(layers_pool.index(layer_index))
            
        return last_value
        
    def process_layer_output(self, x, layer_type):
        """
        Some layer returns intermediate results, usually we dont need that
        """
        if layer_type == 'lstm':
            return x[0]

        else:
            return x

def recalculate_shapes(structure):
    # pool of layers, which should be initialised and connected
    layers_pool = [0]

    # pool of initialised layers
    layers_pool_inited = {}

    # pool of broken (invalid) layers) such as inconsistent number of dimensions
    layers_pool_removed = []

    while layers_pool:
        # take first layer in a pool
        layer_index = layers_pool[0]

        # find all connections before this layer
        enter_layers = set(np.where(structure.matrix[:, layer_index] == 1)[0])

        # check if some of previous layers were not initialized
        # that means - we should initialise them first
        not_inited_layers = [i for i in enter_layers if i not in (layers_pool_inited.keys())]
        not_inited_layers_selected = [layer for layer in not_inited_layers if layer not in layers_pool_removed]

        if not_inited_layers_selected:
            # remove layers, which are in pool already
            # this is possible due to complex connections with different orders
            not_inited_layers_selected = [layer for layer in not_inited_layers_selected if layer not in layers_pool]

            # add not initialised layers to the pool
            layers_pool.extend(not_inited_layers_selected)

            # current layer should be shift to the end of the queue
            acc = layers_pool.pop(0)
            layers_pool.append(acc)
            continue

        # take Layer instance of the previous layers
        input_layers = [structure.layers_index_reverse[layer] for layer in enter_layers]

        # layer without rank is broken and we ignore that
        input_layers = [layer for layer in input_layers if layer.config.get('rank', False)]
        enter_layers = [i for i in enter_layers if i not in layers_pool_removed]

        # if curent layer is the Input - initialise without any input connections
        if not input_layers and structure.layers_index_reverse[layer_index].layer_type == 'input':
            inited_layer = (None, None, None)

        # detect hanging node - some of mutations could remove connection to the layer
        elif not input_layers:
            layers_pool_removed.append(layers_pool.pop(0))
            continue

        # if there are multiple input connections
        elif len(input_layers) > 1:

            # this case does not require additional processing - all logic is inside Layer instance,
            # which handles multiple connections
            inited_layer = structure.layers_index_reverse[layer_index]([None for _ in range(len(input_layers))], input_layers, init=False)
        else:
            input_layers_inited = [layers_pool_inited[layer] for layer in enter_layers][0]
            inited_layer = structure.layers_index_reverse[layer_index](None, input_layers[0], init=False)

        # add new initialised layer
        layers_pool_inited[layer_index] = inited_layer

        # find outgoing connections and add them to the pool
        output_layers = [layer for layer in np.where(structure.matrix[layer_index] == 1)[0]
                            if layer not in layers_pool and layer not in layers_pool_inited.keys()]

        layers_pool.extend(output_layers)

        # remove current layer from the pool
        layers_pool.pop(layers_pool.index(layer_index))
