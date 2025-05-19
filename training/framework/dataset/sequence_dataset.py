import torch.utils.data
from .text.tokenizers.vocabulary import Vocabulary

class SequenceDataset(torch.utils.data.Dataset):
    in_vocabulary: Vocabulary
    out_vocabulary: Vocabulary
