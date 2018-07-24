from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from .base_processing import ProcessingBase


class ProcessingText(ProcessingBase):
    """
    Data processing class for textual data
    """
    @staticmethod
    def data(x_raw, y_raw, data_processing, create_tokens):
        vocabular = data_processing['vocabular']
        sentences_length = data_processing['sentences_length']
        sequences = x_raw

        if create_tokens:
            tokenizer = Tokenizer(num_words=vocabular)
            tokenizer.fit_on_texts(x_raw)
            sequences = tokenizer.texts_to_sequences(x_raw)

        x = pad_sequences(sequences, sentences_length)
        y = to_categorical(y_raw, num_classes=data_processing['classes'])

        return x, y
