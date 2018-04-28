from collections import namedtuple
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Activation, Bidirectional, Conv1D, Conv2D, Dense, Dropout,
    Embedding, Flatten, GlobalMaxPooling1D, Input,
    RepeatVector)
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import multi_gpu_model, to_categorical

import faker


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
        'vocabular': [5000, 8000, 10000, 15000, 20000, 25000, 30000],
        'sentences_length': [10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150],
        'embedding_dim': [64, 128, 200, 300],
        'trainable': [False, True]}}

LAYERS_POOL = {
    'lstm': {
        'units': [1, 2, 4, 8, 12, 16], 
        'recurrent_dropout': [FLOAT32(i / 100) for i in range(5, 95, 5)],
        'activation': ['tanh', 'relu'],
        'implementation': [1, 2]},

    'cnn': {
        'filters': [4, 8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7],
        'strides': [1, 2, 3],
        'padding': ['valid'],
        'activation': ['tanh', 'relu']},

    'dense': {
        'units': [16, 64, 128, 256, 512],
        'activation': ['softmax', 'sigmoid']},

    'dropout': {'rate': [FLOAT32(i / 100) for i in range(5, 95, 5)]}}

POOL_SIZE = len(LAYERS_POOL)

Event = namedtuple('event', ['type', 'stage'])
fake = faker.Faker()

class Layer():
    """
    Single layer class with compability checking
    """
    def __init__(self, layer_type, previous_layer=None, next_layer=None, classes=None):
        self.type = layer_type
        self.classes = classes
        self.config = {}

        self._init_parameters_()
        self._check_compability_(previous_layer, next_layer)
        

    def _init_parameters_(self):
        if self.type == 'embedding':
            variables = list(SPECIAL[self.type])
            for parameter in variables:
                self.config[parameter] = np.random.choice(SPECIAL[self.type][parameter])

        elif self.type == 'last_dense':
            variables = list(LAYERS_POOL['dense'])
            for parameter in variables:
                self.config[parameter] = np.random.choice(LAYERS_POOL['dense'][parameter])

        else:
            variables = list(LAYERS_POOL[self.type])
            for parameter in variables:
                self.config[parameter] = np.random.choice(LAYERS_POOL[self.type][parameter])


    def _check_compability_(self, previous_layer, next_layer):
        """
        Check data shape in specific case such as lstm or bi-lstm
        """
        if self.type == 'lstm':
            if next_layer is not None and next_layer != 'Dense':
                self.config['return_sequences'] = True
            else:
                self.config['return_sequences'] = False

        elif self.type == 'bi':
            if next_layer is not None and next_layer != 'Dense':
                self.config['return_sequences'] = True
            else:
                self.config['return_sequences'] = False

        elif self.type == 'last_dense':
            self.config['units'] = self.classes
    

class Individ():
    """
    Invidiv class for text data types
    TODO: add support for different data types
    TODO: add support for different task types
    """
    def __init__(self, stage, data_type='text', task_type='classification', classes=2, parents=None):
        """
        Create individ randomly or with its parents
        parents: set of two individ objects
        """
        self.layers_number = 0
        self.data_type = data_type
        self.task_type = task_type
        self.name = fake.name().replace(' ', '_') + '_' + str(stage)
        self.classes = classes
        self.stage = stage
        self.history = [Event('Init', stage)]
        self.architecture = []
        self.parents = parents

        if self.parents is None:
            self._random_init_()
            self.training = self._random_init_training_()
            self.data = self._random_init_data_processing()
        else:
            self._init_with_crossing_()


    def _init_layer_(self, layer):
        """
        Return layer according its configs as keras object
        """
        if layer.type == 'lstm':
            layer_tf = LSTM(
                units=layer.config['units'], 
                recurrent_dropout=layer.config['recurrent_dropout'], 
                activation=layer.config['activation'], 
                implementation=layer.config['implementation'],
                return_sequences=layer.config['return_sequences'])

        elif layer.type == 'bi':
            layer_tf = Bidirectional(
                LSTM(
                    units=layer.config['units'], 
                    recurrent_dropout=layer.config['recurrent_dropout'], 
                    activation=layer.config['activation'], 
                    implementation=layer.config['implementation'],
                    return_sequences=layer.config['return_sequences']))

        elif layer.type == 'dense':
            layer_tf = Dense(
                units=layer.config['units'], 
                activation=layer.config['activation'])

        elif layer.type == 'last_dense':
            layer_tf = Dense(
                units=layer.config['units'], 
                activation=layer.config['activation'])

        elif layer.type == 'cnn':
            layer_tf = Conv1D(
                filters=layer.config['filters'], 
                kernel_size=[layer.config['kernel_size']], 
                strides=[layer.config['strides']], 
                padding=layer.config['padding'], 
                activation=layer.config['activation'])

        elif layer.type == 'dropout':
            layer_tf = Dropout(rate=layer.config['rate'])

        elif layer.type == 'embedding':
            layer_tf = Embedding(
                input_dim=layer.config['vocabular'],
                output_dim=layer.config['embedding_dim'],
                input_length=layer.config['sentences_length'],
                trainable=layer.config['trainable'])

        return layer_tf


    def _random_init_(self):
        """
        At first, we set probabilities pool and the we change 
        this uniform distribution according to previous layer
        """
        if self.architecture:
            self.architecture = []
        self.layers_number = np.random.randint(1, 10)

        probabilities_pool = np.full((POOL_SIZE), 1 / POOL_SIZE)
        pool_index = {i: name for i, name in enumerate(LAYERS_POOL.keys())}
        previous_layer = None
        next_layer = None
        tmp_architecture = []

        # Create structure
        for i in range(self.layers_number):
            tmp_layer = np.random.choice(list(pool_index.keys()), p=probabilities_pool)
            tmp_architecture.append(tmp_layer)
            probabilities_pool[tmp_layer] *= 2
            probabilities_pool /= probabilities_pool.sum()

        # tmp_architecture = [np.random.choice(list(pool_index.keys()), 
        # p=probabilities_pool) for i in range(self.layers_number)]

        for i, name in enumerate(tmp_architecture):
            if i != 0:
                previous_layer = pool_index[tmp_architecture[i - 1]]
            if i < len(tmp_architecture) - 1:
                next_layer = pool_index[tmp_architecture[i + 1]]
            if i == len(tmp_architecture) - 1:
                next_layer = 'last_dense'
                
            layer = Layer(pool_index[name], previous_layer, next_layer)
            self.architecture.append(layer)

        if self.data_type == 'text':
            # Push embedding for texts
            layer = Layer('embedding')
            self.architecture.insert(0, layer)
        else:
            raise Exception('Unsupported data type')

        if self.task_type == 'classification':
            #Add last layer
            layer = Layer('last_dense', classes=self.classes)
            self.architecture.append(layer)
        else:
            raise Exception('Unsupported task type')


    def _init_with_crossing_(self):
        """
        New individ parameters according its parents (only 2 now, classic)
        TODO: add compability checker after all crossing
        """
        father = self.parents[0]
        mother = self.parents[1]
        # father_architecture - chose architecture from first individ and text and train from second
        # father_training - only training config from first one
        # father_arch_layers - select overlapping layers and replace parameters from the first architecture 
        # with parameters from the second
  
        pairing_type = np.random.choice(['father_architecture', 'father_training', 'father_architecture_layers', 'father_architecture_parameter', 'father_data_processing'])
        self.history.append(Event('Birth', self.stage))
        
        if pairing_type == 'father_architecture':
            # Father's architecture and mother's training and data
            self.architecture = father.architecture
            self.training = mother.training
            self.data = mother.data

        elif pairing_type == 'father_training':
            # Father's training and mother's architecture and data
            self.architecture = mother.architecture
            self.training = father.training
            self.data = mother.data

        elif pairing_type == 'father_architecture_layers':
            # Select father's architecture and replace random layer with mother's layer
            self.architecture = father.architecture
            changes_layer = np.random.choice([i for i in range(1, len(self.architecture) - 1)])
            alter_layer = np.random.choice([i for i in range(1, len(mother.architecture) - 1)])

            self.architecture[changes_layer] = mother.architecture[alter_layer]
            self.training = father.training
            self.data = father.data

        elif pairing_type == 'father_architecture_parameter':
            # Select father's architecture and change layer parameters with mother's layer
            # dont touch first and last elements - embedding and dense(3), 
            # too many dependencies with text model
            # select common layer
            intersections = set(list(father.architecture[1:-1])) & set(list(mother.architecture[1:-1]))
            intersected_layer = np.random.choice(intersections)
            self.architecture = father.architecture
            
            def find(lst, key, value):
                """
                Return index of element in the list of dictionaries that is equal
                to some value by some key
                """
                for i, dic in enumerate(lst):
                    if dic[key] == value:
                        return i
                return -1

            changes_layer = find(father.architecture, 'name', intersected_layer)
            alter_layer = find(mother.architecture, 'name', intersected_layer)

            self.architecture[changes_layer] = mother.architecture[alter_layer]
            self.training = father.training
            self.data = father.data

        elif pairing_type == 'father_data_processing':
            # Select father's data processing and mother's architecture and training
            # change mother's embedding to avoid mismatchs in dimensions
            self.architecture = mother.architecture
            self.training = mother.training
            self.data = father.data
            
            self.architecture[0] = father.architecture[0]


    def _random_init_training_(self):
        """
        Initialize training parameters
        """
        if not self.architecture:
            raise Exception('Not initialized yet')
        
        variables = list(TRAINING)
        training_tmp = {}
        for i in variables:
            training_tmp[i] = np.random.choice(TRAINING[i])
        return training_tmp

    
    def _random_init_data_processing(self):
        """
        Initialize data processing parameters
        """
        if not self.architecture:
            raise Exception('Not initialized yet')

        if self.data_type == 'text':
            data_tmp = {}
            data_tmp['vocabular'] = self.architecture[0].config['vocabular']
            data_tmp['sentences_length'] = self.architecture[0].config['sentences_length']
            data_tmp['classes'] = self.classes

        return data_tmp

        
    def get_schema(self):
        """
        Return network schema
        """
        schema = [(i.type, i.config) for i in self.architecture]

        return schema

    def get_data_processing(self):
        """
        Return data processing parameters
        """
        return self.data


    def init_tf_graph(self):
        """
        Return tensorflow graph from individ architecture
        """
        if not self.architecture:
            raise Exception('Non initialized net')

        network_graph = Sequential()
        # TODO: add different hacks to solve shape conflicts
        previous_shape = None

        for layer in self.architecture:
            if layer.type == 'last_dense':
                if len(previous_shape) == 3:
                    network_graph.add(Flatten())
            network_graph.add(self._init_layer_(layer))
            previous_shape = network_graph.output.shape
        
        if self.training['optimizer'] == 'adam':
            optimizer = adam(
                lr=self.training['optimizer_lr'],
                decay=self.training['optimizer_decay'])
        else:
            optimizer = RMSprop(
                lr=self.training['optimizer_lr'], 
                decay=self.training['optimizer_decay'])

        if self.task_type == 'classification':
            if self.classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'
        else:
            raise Exception('Unsupported task type')

        return network_graph, optimizer, loss


    def mutation(self, stage):
        """
        Darwin was right. Change some part of individ with some probability
        """
        mutation_type = np.random.choice(['architecture', 'train', 'all'], p=(0.45, 0.45, 0.1))
        self.history.append(Event('Mutation', stage))
        
        if mutation_type == 'architecture':
            # all - change the whole net
            mutation_size = np.random.choice(['all', 'part', 'parameters'], p=(0.3, 0.3, 0.4))
            
            if mutation_size == 'all':
                # If some parts of the architecture change, the processing of data must also change
                self._random_init_()
                self.data = self._random_init_data_processing()

            elif mutation_size == 'part':
                # select layer except the first and the last one - embedding and dense(*)
                mutation_layer = np.random.choice([i for i in range(1, len(self.architecture) - 1)])
                
                # find next layer to avoid incopabilities in neural architecture
                next_layer = self.architecture[mutation_layer + 1]
                new_layer = np.random.choice(list(LAYERS_POOL.keys()))
                layer = Layer(new_layer, next_layer=next_layer)

                self.architecture[mutation_layer] = layer

            elif mutation_size == 'parameters':
                # select layer except the first and the last one - embedding and dense(3)
                mutation_layer = np.random.choice([i for i in range(1, len(self.architecture) - 1)])

                # find next layer to avoid incopabilities in neural architecture
                next_layer = self.architecture[mutation_layer + 1]
                new_layer = self.architecture[mutation_layer].type

                self.architecture[mutation_layer] = Layer(new_layer, next_layer=next_layer)
        
        # TODO: decide that to do with that
        # elif mutation_type == 'text':
        #     mutation_size = np.random.choice(['all', 'part'], p=(0.3, 0.7))
        #     if mutation_size == 'all':
        #         print('wop6')
        #         # reinitialize the text config with existing embedding
        #         text_model = init_text(arch[0])
        #     elif mutation_size == 'part':
        #         print('wop7')
        #         mutation_layer = np.random.choice(list(text_param_pool))
        #         new_model = init_text(arch[0])
        #         text_model[mutation_layer] = new_model[mutation_layer]

        elif mutation_type == 'train':
            mutation_size = np.random.choice(['all', 'part'], p=(0.3, 0.7))

            if mutation_size == 'all':
                self.training = self._random_init_training_()

            elif mutation_size == 'part':
                mutation_parameter = np.random.choice(list(TRAINING))
                new_training = self._random_init_training_()
                self.training[mutation_parameter] = new_training[mutation_parameter]

        elif mutation_type == 'all':
            # change the whole individ - similar to death and rebirth
            self._random_init_()
            self.training = self._random_init_training_()
            self.data = self._random_init_data_processing()
            
    
    def crossing(self, other, stage):
        """
        Create new object as crossing between this one and the other
        """
        new_individ = Individ(stage=stage, classes=2, parents=(self, other))
        return new_individ
        