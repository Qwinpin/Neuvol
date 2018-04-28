from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import multi_gpu_model, to_categorical


class Data():
    def __init__(self, x_raw, y_raw, data_type, task_type):
        """
        x_raw: list of input data
        y_raw: list of output data
        data_type: string variable (text, image, timelines)
        task_type: string variable (classification, regression, autoregression)
        TODO: add support for image and timelines data
        TODO: add support for regression and autoregression
        """
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.data_type = data_type
        self.task_type = task_type


    def process_data(self, data_processing):
        """
        Return data for training
        """
        if self.data_type == 'text':
            vocabular = data_processing['vocabular']
            sentences_length = data_processing['sentences_length']

            tokenizer = Tokenizer(num_words=vocabular)
            tokenizer.fit_on_texts(self.x_raw)
            sequences = tokenizer.texts_to_sequences(self.x_raw)

            x = pad_sequences(sequences, sentences_length)
            y = to_categorical(self.y_raw, num_classes=data_processing['classes'])
        else:
            raise Exception('This data type is not supported now')
            
        return x, y