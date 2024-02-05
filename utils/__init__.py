from typing import Iterator
from typing import Tuple
from typing import Dict
from transformers import BertTokenizer

import torch
import argparse
import logging

from typing import Dict

from utils.post_processor import PostProcessor

import apache_beam as beam

from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerKeyedTensor
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.interactive.display import pipeline_graph
from apache_beam.runners.runner import PipelineResult

from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import BertTokenizer

def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      dest='input',
      help='Path to the text file containing sentences.')
  parser.add_argument(
      '--output',
      dest='output',
      required=True,
      help='Path of file in which to save the output predictions.')
  parser.add_argument(
      '--bert_tokenizer',
      dest='bert_tokenizer',
      default='bert-base-uncased',
      help='bert uncased model. This can be base model or large model')
  parser.add_argument(
      '--large_model',
      action='store_true',
      dest='large_model',
      default=False,
      help='Set to true if your model is large enough to run into memory '
      'pressure if you load multiple copies.')
  parser.add_argument(
      '--job_name',
      action='store_true',
      dest='job_name',
      default="Language Modeling Job",
      help='Job Name.')
  parser.add_argument(
      '--region',
      action='store_true',
      dest='region',
      default="us-west-2",
      help='Region.')
  return parser.parse_known_args(argv)


def add_mask_to_last_word(text: str) -> Tuple[str, str]:
  text_list = text.split()
  return text, ' '.join(text_list[:-2] + ['[MASK]', text_list[-1]])


def filter_empty_lines(text: str) -> Iterator[str]:
  if len(text.strip()) > 0:
    yield text


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

def run(
    argv=None,
    model_class=None,
    model_params=None,
    save_main_session=True,
    test_pipeline=None) -> PipelineResult:
  """
  Args:
    argv: Command line arguments defined for this example.
    model_class: Reference to the class definition of the model.
                If None, BertForMaskedLM will be used as default .
    model_params: Parameters passed to the constructor of the model_class.
                  These will be used to instantiate the model object in the
                  RunInference API.
    save_main_session: Used for internal testing.
    test_pipeline: Used for internal testing.
  """
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
#   pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

  if not model_class:
    model_config = BertConfig.from_pretrained(
        known_args.bert_tokenizer, is_decoder=False, return_dict=True)
    model_class = BertForMaskedLM
    model_params = {'config': model_config}
  
  # TODO: Remove once nested tensors https://github.com/pytorch/nestedtensor
  # is officially released.
  class PytorchNoBatchModelHandler(PytorchModelHandlerKeyedTensor):
    """Wrapper to PytorchModelHandler to limit batch size to 1.

    The tokenized strings generated from BertTokenizer may have different
    lengths, which doesn't work with torch.stack() in current RunInference
    implementation since stack() requires tensors to be the same size.

    Restricting max_batch_size to 1 means there is only 1 example per `batch`
    in the run_inference() call.
    """
    def batch_elements_kwargs(self):
      return {'max_batch_size': 1}

  model_names = ['a', 'b', 'c', 'd', 'e']
  model_handlers = []

  model_name = known_args.bert_tokenizer  # note: in this case, same name works
  # model = BertForMaskedLM.from_pretrained(model_name)
  # torch.save(model.state_dict(), 'model.pth')

  for model_name in model_names:
    model_handlers.append((model_name, PytorchNoBatchModelHandler(
      state_dict_path="gs://dataflow-apache-quickstart_spheric-hawk-126104/models/model.pth",
      model_class=model_class,
      model_params=model_params,
      large_model=known_args.large_model)
    ))
  
  pipeline = test_pipeline
  if not test_pipeline:
    pipeline = beam.Pipeline(options=pipeline_options)

  bert_tokenizer = BertTokenizer.from_pretrained(known_args.bert_tokenizer)

  if not known_args.input:
    text = (pipeline | 'CreateSentences' >> beam.Create([
      'The capital of France is Paris .',
      'It is raining cats and dogs .',
      'He looked up and saw the sun and stars .',
      'Today is Monday and tomorrow is Tuesday .',
      'There are 5 coconuts on this palm tree .',
      'The richest person in the world is not here .',
      'Malls are amazing places to shop because you can find everything you need under one roof .', # pylint: disable=line-too-long
      'This audiobook is sure to liquefy your brain .',
      'The secret ingredient to his wonderful life was gratitude .',
      'The biggest animal in the world is the whale .',
    ]))
  else:
    text = (
        pipeline | 'ReadSentences' >> beam.io.ReadFromText(known_args.input))
  text_and_tokenized_text_tuple = (
      text
      | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines)
      | 'AddMask' >> beam.Map(add_mask_to_last_word)
      | 'TokenizeSentence' >>
      beam.Map(lambda x: tokenize_sentence(x, bert_tokenizer)))
  
  inference_steps = []
  for model_handler in model_handlers:
    name, handler = model_handler
    inference_steps.append((
        text_and_tokenized_text_tuple
        | f'PyTorchRunInference_Model_{name}' >> RunInference(KeyedModelHandler(handler))
    ))

  output = (
      inference_steps | beam.Flatten() 
      | 'ProcessOutput' >> beam.ParDo(PostProcessor(bert_tokenizer=bert_tokenizer)))
  
  output | "WriteOutput" >> beam.io.WriteToText( # pylint: disable=expression-not-assigned
    known_args.output,
    shard_name_template='',
    append_trailing_newlines=True)

  result = pipeline.run()
  result.wait_until_finish()

  digraph = pipeline_graph.PipelineGraph(pipeline).get_dot()

  print(f"\nPipeline Digraph:\n\n{digraph}\n")

  return result
