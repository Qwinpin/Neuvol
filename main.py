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

import architecture
import evaluation


def main():
    df = pd.read_csv('example.csv')
    x_tmp = df.x.astype(str).tolist()
    y_tmp = df.y.astype(int).tolist()

    print(len(x_tmp), len(y_tmp))
    # Create objects
    options = {'classes': 3}
    ind = architecture.Individ(stage=1, data_type='text', task_type='classification', parents=None, **options)
    ev = evaluation.Evaluator(x_tmp, y_tmp, kfold_number=2, device='cpu', generator=False)

    # Set evaluation parameters
    ev.set_verbose(level=1)

    # Show architecture
    print(ind.get_schema())

    # Random mutation
    print('\n\nMutation\n\n')
    ind.mutation(stage=2)

    # Show again
    print(ind.get_schema())

    # Show his story and name
    print(ind.get_history(), ind.get_name())

    # Train this model
    result = ev.fit(network=ind)

    # Show result as AUC score (default). One value for each class
    print('AUC: ', result)

if __name__ == "__main__":
    main()
