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
import copy
from functools import wraps
import json
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np


# in general, json does not handle numpy types
class Custom_Encoder(json.JSONEncoder):
    """
    Custom encoder with numpy handling
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)

        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def dump(data, file_name):
    """
    Save chosen object to the file
    """
    with open(file_name, 'w') as output:
        json.dump(data, output, cls=Custom_Encoder)


def load(file_name):
    """
    Load object from the file
    """
    with open(file_name, 'rb') as inp:
        data = json.load(inp)

    return data


def parameters_copy(func):
    @wraps(func)
    def wrapper(*args):
        copies = []
        for arg in args:
            copies.append(copy.deepcopy(arg))

        return func(*copies)

    return wrapper


def visualize_graph(model, file=None):
    if file is None:
        return model_to_dot(model.init_tf_graph_cycle()).create(prog='dot', format='png')

    else:
        with open('{}.png'.format(file), 'wb') as f:
            f.write(model_to_dot(model.init_tf_graph_cycle()).create(prog='dot', format='png'))
