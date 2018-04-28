import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import backend as K
K._LEARNING_PHASE = tf.constant(0)

import time

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

        self.early_stopping = {
            'min_delta': 0.005,
            'patience': 5}
        self.generator = generator
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
                nn, optimizer, loss = network.init_tf_graph()
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

            tf.reset_default_graph()

        training_time -= time.time()

        result = self.test(predicted_out, real_out, network.get_classes())

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
                    workers=2,
                    use_multiprocessing=True)

                predicted = nn.predict_generator(
                    test_generator,
                    workers=2,
                    use_multiprocessing=True)

                #TODO: Generator should work with indexes, so, self.y will be a list of indexes in future
                real = self.y[test]
                
                # Dear Keras, please, study the resources management!
                K.clear_session()

                predicted_out.extend(predicted)
                real_out.extend(real)

            tf.reset_default_graph()

        training_time -= time.time()

        result = self.test(predicted_out, real_out, network.get_classes())

        return result


    def test(self, predicted_out, real_out, classes):
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
                except:
                    fpr[i], tpr[i] = np.zeros(len(real_out)), np.zeros(len(predicted_out))
                roc_auc.append(auc(fpr[i], tpr[i]))
        
            return roc_auc

        else:
            raise Exception('Unrecognized fitness measure')
