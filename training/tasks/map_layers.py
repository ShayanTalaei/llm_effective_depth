from framework.task import task, args, SimpleTask
import framework
from typing import Tuple, Dict, Any
from framework.interfaces.result import LossOnlyResult
import torch
from framework import dataset
from framework.dataset.text.tokenizers import HuggingfaceVocabulary
from torch.nn import functional as F
from tqdm import tqdm

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-map_layers.src", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("-map_layers.tgt", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("-map_layers.src_layer", default=0)
    parser.add_argument("-map_layers.quantize", default=False)
    parser.add_argument("-map_layers.diff_only", default=False)


def import_huggingface():
    global AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


@task()
class MapLayers(SimpleTask):
    def __init__(self, helper: framework.helpers.TrainingHelper):
        import_huggingface()
        super().__init__(helper)

    def create_model(self):
        if self.helper.args.map_layers.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        self.model2 = AutoModelForCausalLM.from_pretrained(self.helper.args.map_layers.tgt, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
        self.model1 = AutoModelForCausalLM.from_pretrained(self.helper.args.map_layers.src, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

        if not self.helper.args.map_layers.quantize:
            self.model2 = self.model2.to(self.helper.device)
            self.model1 = self.model1.to(self.helper.device)

        print(f"The source model has {len(self.model1.model.layers)} layers")
        print(f"The target model has {len(self.model2.model.layers)} layers")

        self.model1.eval()
        self.model2.eval()

        if self.model1.__class__ != self.model2.__class__:
            raise ValueError("Model classes do not match")

        self.dmodel_1 = self.model1.model.embed_tokens.weight.shape[1]
        self.dmodel_2 = self.model2.model.embed_tokens.weight.shape[1]

        self.layer_logs = []

        def layer_hook(module, input, output):
            if not self.helper.args.map_layers.diff_only:
                self.layer_logs.append(output[0])
            else:
                self.layer_logs.append(output[0] - input[0])

        def model_pre_hook(module, input):
            self.layer_logs.clear()
            if not self.helper.args.map_layers.diff_only:
                self.layer_logs.append(input[0])

        def result_hook(module, input, output):
            self.result = output[0] - input[0]

        model = torch.nn.Linear(self.dmodel_1, self.dmodel_2 * (len(self.model2.model.layers)+int(not self.helper.args.map_layers.diff_only)))

        self.model1.model.layers[0]._forward_pre_hooks.clear()
        self.model1.model.layers[0].register_forward_pre_hook(model_pre_hook)
        self.model1.model.norm = torch.nn.Identity()
        self.model1.lm_head = torch.nn.Identity()

        self.model2.model.norm = torch.nn.Identity()
        self.model2.lm_head = torch.nn.Identity()

        for i in range(len(self.model1.model.layers)-1, self.helper.args.map_layers.src_layer-1, -1):
            del self.model1.model.layers[i]

        print("Training from output of layer ", len(self.model1.model.layers)-1)

        self.model2.model.layers[0]._forward_pre_hooks.clear()
        self.model2.model.layers[0].register_forward_pre_hook(model_pre_hook)

        for i in range(len(self.model2.model.layers)):
            self.model2.model.layers[i].register_forward_hook(layer_hook)

        if self.helper.args.map_layers.diff_only:
            self.model1.model.layers[-1].register_forward_hook(result_hook)

        return model

    def create_datasets(self):
        self.batch_dim = 1

        tokenizer1 = AutoTokenizer.from_pretrained(self.helper.args.map_layers.src, use_fast=True)
        tokenizer2 = AutoTokenizer.from_pretrained(self.helper.args.map_layers.tgt, use_fast=True)

        vocabulary = HuggingfaceVocabulary(tokenizer1)

        if self.helper.args.stop_after is not None:
            train_token_limit = self.helper.args.lm.unroll * self.helper.args.batch_size * (self.helper.args.stop_after + 100)
        else:
            train_token_limit = None

        if tokenizer1.get_vocab() != tokenizer2.get_vocab():
            raise ValueError("Tokenizers do not match")

        self.train_set = dataset.C4(
            self.helper.args.lm.unroll-1, split="train", n_tokens=self.helper.args.sentencepiece.n_pieces,
            token_limit=train_token_limit, vocabulary=vocabulary)
        self.valid_sets.val = dataset.C4(
            self.helper.args.lm.unroll-1, split="validation", n_tokens=self.helper.args.sentencepiece.n_pieces,
            token_limit=self.helper.args.lmds.n_validation_tokens, vocabulary=vocabulary)


    def get_src_target(self, data):
        data = F.pad(data["data"][:-1].T, (1, 0), value=self.train_set.vocabulary.bos_token())
        with torch.no_grad():
            res = self.model1(input_ids=data)
            self.model2(input_ids=data)

            target = torch.stack(self.layer_logs, dim=-2)
            self.layer_logs.clear()

            # Do not include the BOS token in every batch
            if not self.helper.args.map_layers.diff_only:
                src = res.logits[:, 1:]
            else:
                src = self.result[:, 1:]

            target = target[:, 1:]
            del res

            if self.helper.args.map_layers.diff_only:
                del self.result
                self.result = None

        return src, target

    def run_model(self, data, ubatch: int) -> Tuple[LossOnlyResult, Dict[str, Any]]:
        src, target = self.get_src_target(data)

        res = self.model(src)
        del src
        src = None

        loss = F.mse_loss(res, target.flatten(start_dim=-2))
        return LossOnlyResult(loss), {}

    def validate(self):
        res_dict = {}
        self.model.eval()

        for name, loader in self.valid_loaders.items():
            loss_accu = 0
            norm_accu = 0
            err_norm_accu = 0
            loss_count = 0
            relative_accu = 0
            for d in tqdm(loader, desc=f"Validating {name}"):
                with torch.no_grad():
                    with torch.amp.autocast(self.helper.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16 if self.bf16_enabled else None):
                        d = self.prepare_data(d)
                        src, target = self.get_src_target(d)
                        res = self.model(src)
                        loss = F.mse_loss(res.view_as(target), target, reduction="none").permute(2,0,1,3).flatten(start_dim=1).mean(dim=-1)
                        loss_accu += loss
                        tnorm = target.norm(dim=-1).flatten(end_dim=-2)
                        err = (res.view_as(target) - target).norm(dim=-1).flatten(end_dim=-2)
                        norm_accu += tnorm.mean(0)
                        err_norm_accu += err.mean(0)
                        relative_accu += (err / tnorm.clamp(min=torch.finfo(tnorm.dtype).tiny)).mean(0)
                        loss_count += 1

            loss_accu /= loss_count
            err_norm_accu /= norm_accu
            relative_accu /= loss_count
            for i, v in enumerate(loss_accu):
                res_dict[f"{name}/layer_{i}"] = v.item()
                res_dict[f"{name}/norm/layer_{i}"] = err_norm_accu[i].item()
                res_dict[f"{name}/relative/layer_{i}"] = relative_accu[i].item()

        self.model.train()
        return res_dict
