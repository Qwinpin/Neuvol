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
from keras.datasets import imdb

import neuvol


def main():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        path="imdb.npz",
        num_words=30000,
        skip_top=0,
        maxlen=100,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3)

    evaluator = neuvol.Evaluator(x_train, y_train, kfold_number=1)
    mutator = neuvol.Mutator()

    evaluator.create_tokens = False
    evaluator.fitness_measure = 'f1'
    options = {'classes': 2, 'shape': (100,), 'depth': 4}

    wop = neuvol.evolution.Evolution(
                                    stages=10,
                                    population_size=10,
                                    evaluator=evaluator,
                                    mutator=mutator,
                                    data_type='text',
                                    task_type='classification',
                                    active_distribution=True,
                                    freeze=None,
                                    **options)
    wop.cultivate()

    for individ in wop.population_raw_individ:
        print('Architecture: \n')
        print(individ.schema)
        print('\nScore: ', individ.result)


if __name__ == "__main__":
    main()
