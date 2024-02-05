import torch

from typing import Dict
from typing import Iterable
from typing import Tuple

import apache_beam as beam

from apache_beam.ml.inference.base import PredictionResult

from transformers import BertTokenizer

def tokenize_sentence(
    text_and_mask: Tuple[str, str],
    bert_tokenizer: BertTokenizer) -> Tuple[str, Dict[str, torch.Tensor]]:
  text, masked_text = text_and_mask
  tokenized_sentence = bert_tokenizer.encode_plus(
      masked_text, return_tensors="pt")

  # Workaround to manually remove batch dim until we have the feature to
  # add optional batching flag.
  # TODO(https://github.com/apache/beam/issues/21863): Remove once optional
  # batching flag added
  return text, {
      k: torch.squeeze(v)
      for k, v in dict(tokenized_sentence).items()
  }