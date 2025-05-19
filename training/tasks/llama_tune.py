from framework import dataset
from framework.task import task, args
import framework
from .helpers import LMTaskMixin, QuantizedLengthTransformer
import torch

import torch.nn
import torch.utils.data
from framework.task import args, task, SimpleTask
import framework
from framework import dataset
from typing import Dict, Any, Tuple


datasets = None
transformers = None

def load_huggingface():
    global datasets
    global transformers

    import datasets
    import transformers

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-hf.dataset", default="openwebtext")
    parser.add_argument("-hf.model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-hf.n_valid_tokens", default=2_000_000)
    parser.add_argument("-hf.target_only", default=False)


def huggingface_monkeypatch(model: torch.nn.Module):
    # Monkeypatching the model to only save and load the trained parameters
    def get_trainable_params(self):
        return {n for n, p in model.__class__.named_parameters(model) if torch.is_tensor(p) and p.requires_grad}

    def state_dict(self):
        trained_params = get_trainable_params(self)
        return {k: v for k, v in model.__class__.state_dict(self).items() if k in trained_params}

    def load_state_dict(self, state_dict, strict=True):
        old_state = model.__class__.state_dict(self)
        trained_params = get_trainable_params(self)
        mixed_dict = {k: state_dict[k] if k in trained_params else v for k, v in old_state.items()}
        return model.__class__.load_state_dict(self, mixed_dict, strict)

    def named_parameters(self):
        trained_params = get_trainable_params(self)
        return ((n, p) for n, p in model.__class__.named_parameters(self) if n in trained_params)

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    model.state_dict = state_dict.__get__(model)
    model.load_state_dict = load_state_dict.__get__(model)
    model.parameters = parameters.__get__(model)
    model.named_parameters = named_parameters.__get__(model)


@task()
class HuggingfaceTune(QuantizedLengthTransformer, LMTaskMixin, framework.task.SimpleTask):
    PAD_QUANTUM = 64
    NO_OUTPUT_TRACKING = True
    helper: framework.helpers.TrainingHelper
    model: torch.nn.Module
    train_set: dataset.SequenceDataset

    def __init__(self, helper: framework.helpers.TrainingHelper):
        framework.task.SimpleTask.__init__(self, helper)
        LMTaskMixin.__init__(self, loss_mask_name="eval_mask" if self.helper.args.hf.target_only else None)

    def load_base_model(self):
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.helper.args.hf.model,
            torch_dtype=torch.bfloat16,
        )

    def model_post_compile(self, model: torch.nn.Module) -> torch.nn.Module:
        huggingface_monkeypatch(model)
        return model

    def create_model(self) -> torch.nn.Module:
        load_huggingface()

        torch.set_float32_matmul_precision('high')

        model = self.load_base_model()
        model.config.use_cache = False

        return model

    def create_tokenizer(self):
        load_huggingface()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.helper.args.hf.model, fast=True)

    def _run_model(self, input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        res = self.model(input)
        return res.logits, {}

    def create_datasets(self):
        self.create_tokenizer()
        self.batch_dim = 1

        n_valid_tokens = self.helper.args.hf.n_valid_tokens
        n_tok = self.helper.args.lm.unroll * self.helper.args.batch_size * self.helper.args.stop_after + n_valid_tokens
        ds = dataset.HuggingfaceLMDataset(self.helper.args.hf.dataset, self.tokenizer, self.helper.args.lm.unroll, n_tok)

        if n_valid_tokens == 0:
            self.train_set = ds
        else:
            self.valid_sets.valid, self.train_set = dataset.transformations.chunk_dataset(ds, [n_valid_tokens // self.helper.args.lm.unroll, None])

        super().create_datasets()
