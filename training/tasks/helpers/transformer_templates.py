from framework.task import task, args
from framework.helpers import TrainingHelper
import framework
from layers import MoEUTLayer, LanguageModel, MoEUT, PreLNTransformerLayer, RopeAttention, TransformerFFN, Transformer
from .lm_mixin import LMTaskMixin
from .qunatized_length_transformer import QuantizedLengthTransformer
import torch


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-state_size", default=512)
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-transformer.n_layers", default=12)
    parser.add_argument("-transformer.ff_multiplier", default=4.0)
    parser.add_argument("-transformer.universal.group_size", default=2)
    parser.add_argument("-transformer.n_heads", default=8)
    parser.add_argument("-transformer.head_projection_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.unroll", default=512)
    parser.add_argument("-moe.expert_size", default=128)
    parser.add_argument("-moe.n_experts", default=128)
    parser.add_argument("-moe.att.n_experts", default=8)
    parser.add_argument("-moe.k", default=8)
    parser.add_argument("-moe.mlp_sel", default=False)
    parser.add_argument("-tied_embedding", default=False)
    parser.add_argument("-sentencepiece.n_pieces", default=8000)
    parser.add_argument("-lmds.n_validation_tokens", default=2000000)


class MoEUTTask(QuantizedLengthTransformer, LMTaskMixin, framework.task.SimpleTask):
    PAD_QUANTUM = 64

    def __init__(self, helper: TrainingHelper):
        framework.task.SimpleTask.__init__(self, helper)
        LMTaskMixin.__init__(self)

    def create_layer(self) -> torch.nn.Module:
        return MoEUTLayer(
            d_model=self.helper.args.state_size,
            n_heads=self.helper.args.transformer.n_heads,
            ff_expert_size=self.helper.args.moe.expert_size,
            ff_n_experts=self.helper.args.moe.n_experts,
            att_n_experts=self.helper.args.moe.att.n_experts,
            att_proj_size=self.helper.args.transformer.head_projection_size,
            ff_k=self.helper.args.moe.k,
            dropout=self.helper.args.dropout
        )

    def create_inner_model(self) -> torch.nn.Module:
        return MoEUT(
            self.create_layer,
            d_model=self.helper.args.state_size,
            n_layers=self.helper.args.transformer.n_layers,
            group_size=self.helper.args.transformer.universal.group_size,
        )

    def create_model(self) -> torch.nn.Module:
        self.validation_started_on = None
        return LanguageModel(
            self.create_inner_model(),
            n_tokens=len(self.train_set.vocabulary),
            d_model=self.helper.args.state_size,
            n_layers=self.helper.args.transformer.n_layers,
            tied=self.helper.args.tied_embedding,
        )



class TransformerTask(QuantizedLengthTransformer, LMTaskMixin, framework.task.SimpleTask):
    PAD_QUANTUM = 64

    def __init__(self, helper: TrainingHelper):
        framework.task.SimpleTask.__init__(self, helper)
        LMTaskMixin.__init__(self)

    def create_layer(self) -> torch.nn.Module:
        return PreLNTransformerLayer(
            attention=RopeAttention(
                state_size=self.helper.args.state_size,
                n_heads=self.helper.args.transformer.n_heads,
                projection_size=self.helper.args.transformer.head_projection_size
            ),
            ffn=TransformerFFN(
                d_model=self.helper.args.state_size,
                d_ff=int(self.helper.args.state_size * self.helper.args.transformer.ff_multiplier),
                d_out = self.helper.args.state_size,
            ),
            d_model=self.helper.args.state_size,
        )

    def create_inner_model(self) -> torch.nn.Module:
        return Transformer(
            self.create_layer,
            n_layers=self.helper.args.transformer.n_layers,
        )

    def create_model(self) -> torch.nn.Module:
        self.validation_started_on = None
        model = LanguageModel(
            self.create_inner_model(),
            n_tokens=len(self.train_set.vocabulary),
            d_model=self.helper.args.state_size,
            n_layers=self.helper.args.transformer.n_layers,
            tied=self.helper.args.tied_embedding
        )
        return model
