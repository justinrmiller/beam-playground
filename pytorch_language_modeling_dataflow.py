#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""""A pipeline that uses RunInference to perform Language Modeling with Bert.

This pipeline takes sentences from a custom text file, converts the last word
of the sentence into a [MASK] token, and then uses the BertForMaskedLM from
Hugging Face to predict the best word for the masked token given all the words
already in the sentence. The pipeline then writes the prediction to an output
file in which users can then compare against the original sentence.

python3 \
    pytorch_language_modeling_dataflow.py \
    --setup_file=./setup.py \
    --region us-west2 \
    --input \
    gs://dataflow-apache-quickstart_spheric-hawk-126104/language-modeling/input/duplicated_input.txt \
    --output \
    gs://dataflow-apache-quickstart_spheric-hawk-126104/language-modeling/output \
    --runner DataflowRunner \
    --project spheric-hawk-126104 \
    --machine_type n2-custom-8-18432 \
    --temp_location \
    gs://dataflow-apache-quickstart_spheric-hawk-126104/language-modeling/temp/
"""

import logging

from utils import run


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
