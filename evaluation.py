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
import time
import numpy as np
import tensorflow as tf
from keras import backend as K

K._LEARNING_PHASE = tf.constant(0)

from keras.callbacks import EarlyStopping
from sklearn.metrics import (f1_score)
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from data_processing import DataGenerator
from data_processing import Data


class Evaluator():
    """
    Train network and evaluate
    All these parameters - metaparameters of training
    Set it before you start if you want
    """

    def __init__(self, x, y, kfold_number=5, device='cpu', generator=False):
        """
        device available value:
            - gpu
            - cpu
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.kfold_number = kfold_number

        if device == 'cpu':
            self.device = '/device:CPU:0'
        elif device == 'gpu':
            self.device = '/device:GPU:0'
        self.generator = generator

        self.use_multiprocessing = True
        self.workers = 2
        self.early_stopping = {
            'min_delta': 0.005,
            'patience': 5}
        self.verbose = 0
        self.fitness_measure = 'AUC'

    def set_early_stopping(self, min_delta=0.005, patience=5):
        """
        Set early stopping parameters for training
        Please, be careful
        """

        self.early_stopping['min_delta'] = min_delta
        self.early_stopping['patience'] = patience

    def set_verbose(self, level=0):
        """
        Set verbose level, dont touch it, with large amount of individs the number
        of info messages will be too large
        """
        self.verbose = level

    def set_fitness_measure(self, measure):
        """
        Set fitness measure - This parameter determines the criterion
        for the effectiveness of the model
        measure available values:
            - AUC
            - f1
        """
        self.fitness_measure = measure

    def set_device(self, device='cpu', number=1):
        """
        Manualy device management
        device: str cpu or gpu
        number: int number of available devices
        memory_limit: int size of available memory
        TODO: set multiple devices, set memory limits
        """
        if device == 'cpu':
            self.device == '/device:CPU:' + str(number)
        elif device == 'gpu':
            self.device == '/device:GPU:' + str(number)

    def set_DataGenerator_multiproc(self, use_multiprocessing=True, workers=2):
        """
        Set multiprocessing parameters for data generator
        """
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers

    def fit(self, network):
        """
        Training function. N steps of cross-validation
        """
        training_time = time.time()
        predicted_out = []
        real_out = []

        data = Data(
            self.x,
            self.y,
            data_type=network.get_data_type(),
            task_type=network.get_task_type(),
            data_processing=network.get_data_processing())

        x, y = data.process_data()

        kfold = StratifiedKFold(n_splits=self.kfold_number)
        for train, test in kfold.split(np.zeros(x.shape), y.argmax(-1)):
            # work only with this device
            with tf.device(self.device):
                try:
                    nn, optimizer, loss = network.init_tf_graph()
                    print(nn.summary())
                    nn.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        min_delta=self.early_stopping['min_delta'],
                        patience=self.early_stopping['patience'],
                        mode='auto',
                        verbose=False)
                    callbacks = [early_stopping]

                    nn.fit(
                        x[train], y[train],
                        batch_size=network.get_training_parameters()['batchs'],
                        epochs=network.get_training_parameters()['epochs'],
                        validation_data=(x[test], y[test]),
                        callbacks=callbacks,
                        shuffle=True,
                        verbose=self.verbose)

                    predicted = nn.predict(x[test])
                    real = y[test]

                    # Dear Keras, please, study the resources management!
                    K.clear_session()

                    predicted_out.extend(predicted)
                    real_out.extend(real)
                except Exception as e:
                    print(e)
                    return 0.0

            tf.reset_default_graph()

        training_time -= time.time()

        if network.get_task_type() == 'classification':
            result = self.test_classification(predicted_out, real_out, network.options['classes'])

        return result

    def fit_generator(self, network):
        """
        Training function with generators. N steps of cross-validation
        """
        training_time = time.time()
        predicted_out = []
        real_out = []

        kfold = StratifiedKFold(n_splits=self.kfold_number)
        for train, test in kfold.split(np.zeros(len(self.x)), np.zeros(len(self.y))):
            train_generator = DataGenerator(
                self.x[train],
                self.y[train],
                data_type=network.get_data_type(),
                task_type=network.get_task_type(),
                data_processing=network.get_data_processing())

            test_generator = DataGenerator(
                self.x[test],
                self.y[test],
                data_type=network.get_data_type(),
                task_type=network.get_task_type(),
                data_processing=network.get_data_processing())

            # work only with this device
            with tf.device(self.device):
                nn, optimizer, loss = network.init_tf_graph()
                nn.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    min_delta=self.early_stopping['min_delta'],
                    patience=self.early_stopping['patience'],
                    mode='auto',
                    verbose=False)
                callbacks = [early_stopping]

                nn.fit_generator(
                    generator=train_generator,
                    epochs=network.get_training_parameters()['epochs'],
                    verbose=self.verbose,
                    callbacks=callbacks,
                    validation_data=test_generator,
                    workers=self.workers,
                    use_multiprocessing=self.use_multiprocessing)

                predicted = nn.predict_generator(
                    test_generator,
                    workers=self.workers,
                    use_multiprocessing=self.use_multiprocessing)

                # TODO: Generator should work with pointers to files,
                # TODO: so, self.y and self.x will be a list of file names in future
                real = self.y[test]

                # Dear Keras, please, study the resources management!
                K.clear_session()

                predicted_out.extend(predicted)
                real_out.extend(real)

            tf.reset_default_graph()

        training_time -= time.time()

        if network.task_type == 'classification':
            result = self.test_classification(predicted_out, real_out, network.options['classes'])

        return result

    def test_classification(self, predicted_out, real_out, classes):
        """
        Return fitness results
        """
        if self.fitness_measure == 'f1':
            predicted = np.array(predicted_out).argmax(-1)
            real = np.array(real_out).argmax(-1)

            f1 = f1_score(real, predicted, average=None)
            # precision = precision_score(real, predicted, average=None)
            # recall = recall_score(real, predicted, average=None)
            # accuracy = accuracy_score(real, predicted)
            return f1

        elif self.fitness_measure == 'AUC':
            fpr = dict()
            tpr = dict()
            roc_auc = []

            for i in range(classes):
                try:
                    fpr[i], tpr[i], _ = roc_curve(np.array(real_out)[:, i], np.array(predicted_out)[:, i])
                except Exception as e:
                    fpr[i], tpr[i] = np.zeros(len(real_out)), np.zeros(len(predicted_out))
                roc_auc.append(auc(fpr[i], tpr[i]))

            return roc_auc

        else:
            raise Exception('Unrecognized fitness measure')
