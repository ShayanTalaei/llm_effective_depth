from framework.task import task, args
import framework
from framework import dataset
from framework.helpers import TrainingHelper
from .helpers import TransformerTask, MoEUTTask

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-dm_math.filter", default="none", parser=parser.str_or_none_parser)
    parser.add_argument("-dm_math.loss_on_target_only", default=False)


class DMMathMixin:
    def __init__(self, helper: TrainingHelper):
        super().__init__(helper)
        if self.helper.args.dm_math.loss_on_target_only:
            self.loss_mask_name = "eval_mask"

    def create_datasets(self):
        self.batch_dim = 1

        # Magic number for backward compatibility
        self.train_set = dataset.DeepmindMathAutoregressiveDataset(
            self.helper.args.lm.unroll,
            splits=["train-easy", "train-medium", "train-hard"],
            split_filter_regex=self.helper.args.dm_math.filter,
            no_split=True)

        print(f"{self.__class__.__name__}: Loaded {len(self.train_set)} train examples")

        self.valid_sets.interpolate = framework.dataset.transformations.LimitDatasetLength(
            framework.dataset.transformations.PermuteDataset(
                dataset.DeepmindMathAutoregressiveDataset(
                    self.helper.args.lm.unroll, ["interpolate"], no_split=True,
                    split_filter_regex=self.helper.args.dm_math.filter)
            ), 5000)
        self.valid_sets.extrapolate = framework.dataset.transformations.LimitDatasetLength(
            framework.dataset.transformations.PermuteDataset(
                dataset.DeepmindMathAutoregressiveDataset(
                    self.helper.args.lm.unroll, ["extrapolate"], no_split=True,
                    split_filter_regex=self.helper.args.dm_math.filter)
            ), 5000)

        self.prob_compare_valid_sets = framework.data_structures.DotDict()


@task()
class DMMathTransformer(DMMathMixin, TransformerTask):
    pass


@task()
class DMMathMoeut(DMMathMixin, MoEUTTask):
    pass

