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

import pandas as pd
import numpy as np
import random
import string

from neuvol import evolution, evaluation


def main():
    df = pd.read_csv('train.csv', names=['label', 'date', 'qwqw', 'name', 'text'], encoding='latin-1').sample(n=10000)
    x_tmp = df.text.astype(str).tolist()
    y_tmp = df.label.apply(lambda b: 0 if b == 0 else 1).astype(int).tolist()
    print(len(x_tmp), len(y_tmp))
    # Create objects
    options = {'classes': 2}
    ind = architecture.Individ(stage=1, data_type='text', task_type='classification', parents=None, **options)
    ev = evaluation.Evaluator(x_tmp, y_tmp, kfold_number=2, device='cpu', generator=False)

    # Set evaluation parameters
    ev.set_verbose(level=1)

    # Show architecture
    print(ind.get_schema())

    # Random mutation
    print('\n\nMutation\n\n')
    # ind.mutation(stage=2)

    # Show again
    # print(ind.get_schema())

    # Show his story and name
    print(ind.get_history(), ind.get_name())

    # Show shape without initialisation
    print(ind.shape_structure)

    # Train this model
    result = ev.fit(network=ind)

    # Show result as AUC score (default). One value for each class
    print('AUC: ', result)


def main_ev():
    x = [' '.join([''.join(random.sample(string.ascii_lowercase, k=15)) for _ in range(25)]) for _ in range(1000)]
    y = np.random.randint(0, 2, size=(1000)).tolist()

    ev = evaluation.Evaluator(x, y, 1, generator=False)
    options = {'classes': 2}
    wop = evolution.Evolution(10, 5, ev, **options)
    wop.cultivate()

    for ind in wop.population:
        print(ind.result)


if __name__ == "__main__":
    main_ev()
