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
from keras.utils import Sequence
import numpy as np

from .processing_interface import processing


class DataGenerator(Sequence):
    """
    Generate data samples
    """
    def __init__(self, x_raw, y_raw, data_processing, data_type, task_type, batch_size=32, shuffle=True, create_tokens=True):
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.data_processing = data_processing
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_type = data_type
        self.task_type = task_type
        self.create_tokens = create_tokens

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

        X, Y = self._data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        """
        Shuffle data indexes after each epoch
        """
        self.indexes = np.arange(len(self.x_raw))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, indexes):
        """
        Generate data in batchs
        """
        x, y = processing(self.data_type).data(self.x_raw, self.y_raw, self.data_processing, self.create_tokens)

        X = np.empty((self.batch_size, self.data_processing['sentences_length']))
        Y = np.empty((self.batch_size, self.data_processing['classes']), dtype=int)

        for i, index in enumerate(indexes):
            X[i, ] = x[index]
            Y[i, ] = y[index]

        return X, Y


class Data():
    """
    Class for data generation
    """
    def __init__(self, x_raw, y_raw, data_type, task_type, data_processing, create_tokens=True):
        """
        x_raw: list of input data
        y_raw: list of target data
        data_type: string variable (text, image, timelines)
        task_type: string variable (classification, regression, autoregression)
        TODO: add support for image and timelines data
        TODO: add support for regression and autoregression
        """
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.data_type = data_type
        self.task_type = task_type
        self.data_processing = data_processing
        self.create_tokens = create_tokens

    def process_data(self):
        """
        Return data for training
        """
        x, y = processing(self.data_type).data(self.x_raw, self.y_raw, self.data_processing, self.create_tokens)

        return x, y
