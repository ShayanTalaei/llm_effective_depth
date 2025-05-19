
from typing import List, Dict, Any, Optional
from .vocabulary import Vocabulary


class HuggingfaceVocabulary(Vocabulary):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.tokenizer)

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(indices)

    def sentence_to_indices(self, sentence: str) -> List[int]:
        return self.tokenizer(sentence, add_special_tokens=False)["input_ids"]

    def tokenize_batch(self, seq: List[str]) -> List[List[int]]:
        return self.sentence_to_indices(seq)

    def to_string(self, indices: List[int]) -> str:
        return self.tokenizer.decode(indices)

    def id(self) -> str:
        return f"{self.__class__.__name__}({self.escape_path(self.tokenizer.name_or_path)})"

    def allow_multiprocessing(self) -> bool:
        return not self.tokenizer.is_fast

    def bos_token(self) -> Optional[int]:
        return getattr(self.tokenizer, "bos_token_id", None)
