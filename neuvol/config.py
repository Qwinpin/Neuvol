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

import logging

# it is important if you use
import tensorflow as tf


config = tf.compat.v1.ConfigProto()
# config.gpu_options.visible_device_list = "1"
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.allow_soft_placement = True
# config.gpu_options.allow_growth = True
SESSION = tf.compat.v1.Session(config=config)

HANDLER = logging.FileHandler("log.log")
FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')

LOGGER = logging.getLogger('default')
LOGGER.setLevel(logging.INFO)

HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDLER)
