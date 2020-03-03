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
from .processing_interface import processing


class Data():
    """
    Class for data generation
    """
    def __init__(self, x_raw, y_raw, data_type, task_type, data_processing, create_tokens=True):
        """
        x_raw: list of input data
        y_raw: list of target data
        data_type: string variable (text, image, timelines)
        task_type: string variable (classification, regression, autoregression)
        TODO: add support for image and timelines data
        TODO: add support for regression and autoregression
        """
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.data_type = data_type
        self.task_type = task_type
        self.data_processing = data_processing
        self.create_tokens = create_tokens

    def process_data(self):
        """
        Return data for training
        """
        x, y = processing(self.data_type).data(self.x_raw, self.y_raw, self.data_processing, self.create_tokens)

        return x, y
