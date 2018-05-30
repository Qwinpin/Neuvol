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
