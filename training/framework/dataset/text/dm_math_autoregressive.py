from ... import utils
import os
import torch.utils.data
from ...utils import LockFile, GenToIt, download
from .tokenizers.sentencepiece import SentencepieceVocabulary
from .tokenizers.vocabulary import Vocabulary
import numpy as np
import bisect
from typing import Any, Optional, List, Tuple
import random
import re
import multiprocessing
from .logical_inference_lm import GroupLmTestState
from .lm_dataset import WordLevelLanguageModelTestState
from ..fs_cache import get_cached_file


class DeepmindMathAutoregressiveDataset(torch.utils.data.Dataset):
    ADD_SPACES_REGEX = re.compile(r'([^a-zA-Z0-9\s])')
    REMOVE_MULTIPLE_SPACES_REGEX = re.compile(r'\s+')

    def lock(self) -> utils.LockFile:
        return utils.LockFile(os.path.join(self.cache_dir, "dm_math_lock"))

    def download(self):
        with self.lock():
            os.makedirs(self.cache_dir, exist_ok=True)
            if not os.path.isdir(os.path.join(self.cache_dir, "mathematics_dataset-v1.0")):
                print("Downloading Deepmind Mathematics Dataset...")
                if not os.path.isfile(os.path.join(self.cache_dir, "mathematics_dataset-v1.0.tar.gz")):
                    os.makedirs(f"{self.cache_dir}/tmp/", exist_ok=True)

                    download("https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz", f"{self.cache_dir}/tmp/")
                    os.rename(f"{self.cache_dir}/tmp/mathematics_dataset-v1.0", f"{self.cache_dir}/mathematics_dataset-v1.0")

                print("Done.")

    def file_iterator(self, fname: str):
         with open(fname, "r") as f:
            i = 0
            prev = None
            for l in f:
                if i % 2 == 0:
                    prev = l
                else:
                    yield prev.strip(), l.strip()
                i += 1

    def add_spaces_around_non_alphanumeric(self, text):
        text_with_spaces = self.ADD_SPACES_REGEX.sub(r' \1 ', text)
        text_with_spaces = self.REMOVE_MULTIPLE_SPACES_REGEX.sub(' ', text_with_spaces).strip()

        return text_with_spaces

    def get_tokenizer_train_sentences(self):
        N_LINES_PER_FILE = 10000

        dir = f"{self.cache_dir}/mathematics_dataset-v1.0/train-easy"
        for f in os.listdir(dir):
            if "_swr_" in f:
                # Avoid tokenizing sequences of random characters
                continue

            cnt = 0
            for q, a in self.file_iterator(f"{dir}/{f}"):
                cnt += 1
                if cnt > N_LINES_PER_FILE:
                    break

                q = self.add_spaces_around_non_alphanumeric(q)
                a = self.add_spaces_around_non_alphanumeric(a)

                yield f"Q: {q} A: {a} "

    def get_target_file(self, f: str):
        return os.path.join(self.tokenized_dir, f.replace("/", "_"))

    def tokenize_and_write(self, ofile, index_table: List[int], qalist: List[Tuple[str, str]]):
        big_pile = []
        for q, a in qalist:
            big_pile.extend([q, a])

        tokenized = self.vocabulary.tokenize_batch(big_pile)

        size = np.dtype(self.data_dtype).itemsize

        for i in range(0, len(tokenized), 2):
            q = tokenized[i]
            a = tokenized[i+1]

            index_table.append(ofile.tell()//size)
            np.asarray(q, dtype=self.data_dtype).tofile(ofile)
            index_table.append(ofile.tell()//size)
            np.asarray(a, dtype=self.data_dtype).tofile(ofile)


    def tokenize_file(self, f:str):
        index_table = []
        tokenized_name = self.get_target_file(f)

        if os.path.isfile(tokenized_name):
            return tokenized_name

        print(f"{self.__class__.__name__}: Tokenizing {f}...")

        qalist = []
        with open(tokenized_name+".tmp", "wb") as ofile:
            for q, a in self.file_iterator(f"{self.cache_dir}/mathematics_dataset-v1.0/{f}"):
                qalist.append((" Q: "+q+" A:", " "+a))

                if len(qalist) > 1000:
                    self.tokenize_and_write(ofile, index_table, qalist)
                    qalist = []

            if qalist:
                self.tokenize_and_write(ofile, index_table, qalist)

        with open(tokenized_name+".index", "wb") as f:
            np.asarray(index_table, dtype=np.uint32).tofile(f)

        os.rename(tokenized_name+".tmp", tokenized_name)

        return tokenized_name

    def update_data_type(self):
        # Avoid unnecessary copying
        if self.n_tokens >= 2**31 - 1:
            self.data_dtype = np.int64
        elif self.n_tokens >= 2**15 - 1:
            self.data_dtype = np.int32
        elif self.n_tokens >= 2**8:
            self.data_dtype = np.int16
        else:
            self.data_dtype = np.uint8

    def load(self):
        self.data = []
        self.indices = []
        self.offsets = [0]
        self.names = []

        if self.split_filter_regex is not None:
            filter = re.compile(self.split_filter_regex)
            split_filter = lambda x: filter.fullmatch(x) is not None
        else:
            split_filter = lambda x: True

        missing = []
        with LockFile(self.my_cache_dir + "/lock"):
            for split in self.splits:
                dir = f"{self.cache_dir}/mathematics_dataset-v1.0/{split}"
                for f in os.listdir(dir):
                    if not split_filter(f):
                        continue

                    name = f"{split}/{f}"

                    fname = self.get_target_file(name)
                    if not os.path.isfile(fname):
                        missing.append(name)

            if missing:
                print(f"{self.__class__.__name__}: Missing {len(missing)} files: {', '.join(missing)}. Tokenizing them...")
                with multiprocessing.Pool(min(multiprocessing.cpu_count(), len(missing))) as p:
                    p.map(self.tokenize_file, missing)

        for split in self.splits:
            dir = f"{self.cache_dir}/mathematics_dataset-v1.0/{split}"
            for f in os.listdir(dir):
                if not split_filter(f):
                    continue

                name = f"{split}/{f}"
                fname = self.get_target_file(name)

                print(f"{self.__class__.__name__}: Loading {fname}...")

                self.data.append(np.memmap(get_cached_file(fname), dtype=self.data_dtype, mode='r'))
                self.indices.append(np.memmap(get_cached_file(fname+".index"), dtype=np.uint32, mode='r'))

                self.names.append(name)
                self.offsets.append(self.offsets[-1] + len(self.indices[-1]) // 2)

        if not self.data:
            raise ValueError(f"{self.__class__.__name__}: No data found. Please check the split filter regex.")

    def get(self, index: int):
        i = bisect.bisect_right(self.offsets, index) - 1
        offset = index - self.offsets[i]

        q = self.data[i][self.indices[i][offset*2]:self.indices[i][offset*2+1]]
        if offset*2+2 < len(self.indices[i]):
            a = self.data[i][self.indices[i][offset*2+1]:self.indices[i][offset*2+2]]
        else:
            a = self.data[i][self.indices[i][offset*2+1]:]

        bos_token = self.vocabulary.bos_token()
        if bos_token is not None:
            q = np.concatenate([[bos_token], q], dtype=q.dtype)

        return q, a, i

    def __len__(self):
        return self.offsets[-1]

    def fetch_next(self):
        while len(self.data_buffer) < self.unroll_len:
            i = self.rng.randint(0, self.offsets[-1] - 1)
            q, a, _ = self.get(i)

            lex = q.shape[0] + a.shape[0]
            if self.no_split and lex + len(self.data_buffer) > self.unroll_len:
                # If splitting is not allowed, try until we can't write something in the buffer
                if len(self.data_buffer) == 0:
                    continue
                else:
                    break

            self.data_buffer += q.tolist() + a.tolist()
            self.mask_buffer += [0] * (q.shape[0]) + [1] * (a.shape[0])

    def __getitem__(self, index) -> Any:
        if self.is_train:
            if self.rng is None:
                self.rng = random.Random(index)

            self.fetch_next()

            res = {
                "data": self.data_buffer[:self.unroll_len],
                "eval_mask": self.mask_buffer[:self.unroll_len]
            }

            if self.rng.random() < self.buf_reset_probability or self.no_split:
                self.data_buffer = []
                self.mask_buffer = []
            else:
                self.data_buffer = self.data_buffer[self.unroll_len:]
                self.mask_buffer = self.mask_buffer[self.unroll_len:]
        else:
            q, a, o = self.get(index)

            res = {
                "data": np.concatenate([q, a]),
                "eval_mask": [0] * q.shape[0] + [1] * a.shape[0],
                "split_index": o
            }

        res["mask"] = [1] * len(res["data"])

        return {
            k: np.array(v, dtype=self.data_dtype if k=="data" else bool) if not isinstance(v, np.ndarray) else v for k, v in res.items()
        }

    def __init__(self, unroll_len: int, splits=["train-easy", "train-medium", "train-hard"], n_tokens: int = 512, cache_dir="./cache",
                 no_split: bool = False, split_filter_regex: Optional[str] = None, vocabulary: Optional[Vocabulary] = None):
        self.buf_reset_probability = 0.1

        is_train = all("train" in s for s in splits)
        is_valid = all(not ("train" in s) for s in splits)

        if not (is_train ^ is_valid):
            raise ValueError("Splits must be either train or validation.")

        self.is_train = is_train
        self.split_filter_regex = split_filter_regex

        self.n_tokens = n_tokens if vocabulary is None else len(vocabulary)
        self.unroll_len = unroll_len
        self.no_split = no_split
        self.update_data_type()

        self.cache_dir = f"{cache_dir}/{self.__class__.__name__}"
        self.my_cache_dir = f"{cache_dir}/{self.__class__.__name__}/{n_tokens if not vocabulary else vocabulary.id()}"
        self.tokenized_dir = f"{self.my_cache_dir}/tokenized"
        os.makedirs(self.tokenized_dir, exist_ok=True)

        self._sp_model_name = os.path.join(self.my_cache_dir, "tokenizer.model")

        self.download()

        if vocabulary is None:
            with LockFile(self.my_cache_dir + "/lock"):
                self.vocabulary = SentencepieceVocabulary(self._sp_model_name, GenToIt(self.get_tokenizer_train_sentences), n_tokens)
                print(f"{self.__class__.__name__}: Loaded tokenizer.")
        else:
            self.vocabulary = vocabulary

        self.splits = splits

        self.rng = None
        self.load()
        self.data_buffer = []
        self.mask_buffer = []

    def start_test(self) -> WordLevelLanguageModelTestState:
        return GroupLmTestState(self.names, "split_index", mask_key="eval_mask")

