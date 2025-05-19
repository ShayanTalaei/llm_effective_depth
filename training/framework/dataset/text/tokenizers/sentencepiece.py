
import os
from typing import List, Union, Dict, Any, Union, Iterator
import multiprocessing
from .vocabulary import Vocabulary


class SentencepieceVocabulary(Vocabulary):
    def __init__(self, path: str, train_data: Union[str, Iterator], vocab_size: int):
        global spm
        import sentencepiece as spm

        model_file = path + ".model"

        if not os.path.exists(model_file):
            if isinstance(train_data, str):
                spm.SentencePieceTrainer.train(input=train_data, model_prefix=path, vocab_size=vocab_size, split_digits=True, model_type="bpe")
            else:
                spm.SentencePieceTrainer.train(sentence_iterator=train_data, model_prefix=path, vocab_size=vocab_size, split_digits=True, model_type="bpe")

        self.path = path
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_file)
        self.pool = None
        pass

    def __len__(self) -> int:
        return self.tokenizer.get_piece_size()

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        return [self.tokenizer.IdToPiece(i) for i in indices]

    def sentence_to_indices(self, sentence: str) -> List[int]:
        return self.tokenizer.encode_as_ids(sentence)

    def batch_tokenization_start(self):
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())

    def tokenize_batch(self, seq: List[str]) -> List[List[int]]:
        if self.pool is None:
            return [self.sentence_to_indices(s) for s in seq]
        else:
            return self.pool.map(self.sentence_to_indices, seq)

    def tokenize_batch_end(self):
        self.pool.close()
        self.pool.join()
        self.pool = None

    def to_string(self, seq: List[int]) -> str:
        return self.tokenizer.decode_ids(seq)

    def id(self) -> str:
        return f"{self.__class__.__name__}({self.escape_path(self.path)})"
