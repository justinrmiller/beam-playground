import json

from typing import Dict
from typing import Iterable
from typing import Tuple

import apache_beam as beam

from apache_beam.ml.inference.base import PredictionResult

from transformers import BertTokenizer

class PostProcessor(beam.DoFn):
  """Processes the PredictionResult to get the predicted word.

  The logits are the output of the BERT Model. After applying a softmax
  activation function to the logits, we get probabilistic distributions for each
  of the words in BERT’s vocabulary. We can get the word with the highest
  probability of being a candidate replacement word by taking the argmax.
  """
  def __init__(self, bert_tokenizer: BertTokenizer):
    super().__init__()
    self.bert_tokenizer = bert_tokenizer

  def process(self, element: Tuple[str, PredictionResult]) -> Iterable[str]:
    text, prediction_result = element
    inputs = prediction_result.example
    logits = prediction_result.inference['logits']
    mask_token_index = (
        inputs['input_ids'] == self.bert_tokenizer.mask_token_id).nonzero(
            as_tuple=True)[0]
    predicted_token_id = logits[mask_token_index].argmax(axis=-1)
    decoded_word = self.bert_tokenizer.decode(predicted_token_id)

    model_name = element[1].model_id
    print(f"Model Name: {model_name} - Output: {text + ';' + decoded_word}")
    yield json.dumps({
       "model_id": model_name.split(".")[0],
       "text": text,
       "decoded_word": decoded_word
    })
