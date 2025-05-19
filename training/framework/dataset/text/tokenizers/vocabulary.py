
from typing import List, Union, Dict, Any, Union, Optional


class Vocabulary:
    def __len__(self) -> int:
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]):
        pass

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        raise NotImplementedError()

    def sentence_to_indices(self, sentence: str) -> List[int]:
        raise NotImplementedError()

    def __call__(self, seq: Union[List[Union[str, int]], str]) -> List[Union[int, str]]:
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if isinstance(seq, str) or isinstance(seq[0], str):
            return self.sentence_to_indices(seq)
        else:
            return self.indices_to_sentence(seq)

    def batch_tokenization_start(self):
        pass

    def tokenize_batch(self, seq: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    def tokenize_batch_end(self):
        pass

    def to_string(self, seq: List[int]) -> str:
        raise NotImplementedError()

    def id(self) -> str:
        raise NotImplementedError()

    def allow_multiprocessing(self) -> bool:
        return True

    def repr(self) -> str:
        return self.id()

    def escape_path(self, path: str) -> str:
        return path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_").replace(" ", "_")

    def bos_token(self) -> Optional[int]:
        return None