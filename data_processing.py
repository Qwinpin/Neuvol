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

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import multi_gpu_model, to_categorical
from keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    """
    Generate data samples
    """
    def __init__(self, x_raw, y_raw, data_processing, data_type, task_type, batch_size=32, shuffle=True):
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.data_processing = data_processing
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_type = data_type
        self.task_type = task_type

        self.on_epoch_end()


    def __len__(self):
        """
        Total number of batches per epoch
        """
        return int(np.floor(len(self.x_raw) / self.batch_size))


    def __getitem__(self, index):
        """
        Generate one batch
        """
        indexes = self.indexes[(index * self.batch_size):((index + 1) * self.batch_size)]

        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        """
        Shuffle data indexes after each epoch
        """
        self.indexes = np.arange(len(self.x_raw))

        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation(self, indexes):
        """
        Generate data in batchs
        """
        if self.data_type == 'text':
            vocabular = self.data_processing['vocabular']
            sentences_length = self.data_processing['sentences_length']

            tokenizer = Tokenizer(num_words=vocabular)
            tokenizer.fit_on_texts(self.x_raw)
            sequences = tokenizer.texts_to_sequences(self.x_raw)

            x_raw = pad_sequences(sequences, sentences_length)
            y_raw = to_categorical(self.y_raw, num_classes=self.data_processing['classes'])

            X = np.empty((self.batch_size, self.data_processing['sentences_length']))
            Y = np.empty((self.batch_size, self.data_processing['classes']), dtype=int)

            for i, index in enumerate(indexes):
                X[i,] = x_raw[index]
                Y[i,] = y_raw[index]

            
        return X, Y


class Data():
    def __init__(self, x_raw, y_raw, data_type, task_type, data_processing):
        """
        x_raw: list of input data
        y_raw: list of target data
        data_type: string variable (text, image, timelines)
        task_type: string variable (classification, regression, autoregression)
        TODO: add support for image and timelines data
        TODO: add support for regression and autoregression
        """
        if len(x_raw) != len(y_raw):
            raise Exception('Input length does not match the target length')

        self.x_raw = x_raw
        self.y_raw = y_raw
        self.data_type = data_type
        self.task_type = task_type
        self.data_processing = data_processing


    def process_data(self):
        """
        Return data for training
        """
        if self.data_type == 'text':
            vocabular = self.data_processing['vocabular']
            sentences_length = self.data_processing['sentences_length']

            tokenizer = Tokenizer(num_words=vocabular)
            tokenizer.fit_on_texts(self.x_raw)
            sequences = tokenizer.texts_to_sequences(self.x_raw)

            x = pad_sequences(sequences, sentences_length)
            y = to_categorical(self.y_raw, num_classes=self.data_processing['classes'])
        else:
            raise Exception('This data type is not supported now')
            
        return x, y
