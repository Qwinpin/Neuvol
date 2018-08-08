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
