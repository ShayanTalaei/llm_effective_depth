from framework import dataset
from framework.task import task
import framework
import torch

import torch.nn
import torch.utils.data
from framework.task import task
import framework
from framework import dataset
from framework.dataset.text.tokenizers import HuggingfaceVocabulary
from typing import Dict, Any
from .llama_tune import HuggingfaceTune


datasets = None
transformers = None

def load_huggingface():
    global datasets
    global transformers

    import datasets
    import transformers


@task()
class HuggingfaceTuneDMMath(HuggingfaceTune):
    def create_tokenizer(self):
        load_huggingface()
        self.tokenizer = HuggingfaceVocabulary(
        transformers.AutoTokenizer.from_pretrained(self.helper.args.hf.model, fast=True))

    def load_base_model(self):
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.helper.args.hf.model,
            torch_dtype=torch.float32,
        )

    def create_datasets(self):
        self.create_tokenizer()
        self.batch_dim = 1

        # Magic number for backward compatibility
        self.train_set = dataset.DeepmindMathAutoregressiveDataset(
            self.helper.args.lm.unroll,
            splits=["train-easy", "train-medium", "train-hard"],
            split_filter_regex=self.helper.args.dm_math.filter,
            no_split=True,
            vocabulary=self.tokenizer)

        print(f"{self.__class__.__name__}: Loaded {len(self.train_set)} train examples")

        self.valid_sets.interpolate = framework.dataset.transformations.LimitDatasetLength(
            framework.dataset.transformations.PermuteDataset(
                dataset.DeepmindMathAutoregressiveDataset(
                    self.helper.args.lm.unroll, ["interpolate"], no_split=True,
                    split_filter_regex=self.helper.args.dm_math.filter,
                    vocabulary=self.tokenizer)
            ), 5000)
        self.valid_sets.extrapolate = framework.dataset.transformations.LimitDatasetLength(
            framework.dataset.transformations.PermuteDataset(
                dataset.DeepmindMathAutoregressiveDataset(
                    self.helper.args.lm.unroll, ["extrapolate"], no_split=True,
                    split_filter_regex=self.helper.args.dm_math.filter,
                    vocabulary=self.tokenizer)
            ), 5000)

        self.prob_compare_valid_sets = framework.data_structures.DotDict()
