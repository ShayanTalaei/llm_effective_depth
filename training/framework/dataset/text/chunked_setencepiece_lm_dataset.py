# Based on https://huggingface.co/datasets/c4/blob/main/c4.py

from .tokenizers.sentencepiece import SentencepieceVocabulary
from ...utils.download import UrlStream
from ...utils import LockFile, GenToIt
import gzip
import json
from typing import List, Optional, Dict, Any
import numpy as np
import os
import bisect
import time
import torch.multiprocessing as mp
from .lm_dataset import WordLevelLanguageModelTestState
import math
from ..fs_cache import get_cached_file
import io
import time
from .tokenizers.vocabulary import Vocabulary


zstd = None
def read_lines_from_zst(url: str):
    # from https://stackoverflow.com/questions/61067762/how-to-extract-zst-files-into-a-pandas-dataframe
    global zstd
    if zstd is None:
        import zstandard as zstd

    urls = UrlStream(url)

    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    with (
        zstd.open(urls, mode='rb', dctx=DCTX) as zfh,
        io.TextIOWrapper(zfh) as iofh
    ):
        for line in iofh:
            yield line

class ChunkedSentencepieceLMDataset:
    TOKENIZER_N_FILES = 10
    CHUNK_CLOSE_TIMEOUT = 5*60

    def _get_variant_subid(self) -> str:
        if self.external_vocabulary is not None:
            return f"{self.external_vocabulary.id()}"
        else:
            return f"{self.n_tokens}"

    def _get_variant_id(self) -> str:
        return f"{self.__class__.__name__}-{self._get_variant_subid()}"

    def parse(self, line: str) -> str:
        return json.loads(line)["text"]

    def parse_with_sep(self, line: str) -> str:
        txt = self.parse(line)
        if txt:
            return txt + "<STORY_SEP>"
        return txt

    def gzip_line_iterator(self, url: str):
        stream = UrlStream(url)
        print(f"Opening shard {url}, size {stream.size()}")
        for l in gzip.GzipFile(fileobj=stream):
            yield self.parse_with_sep(l.decode("utf-8"))

    def zst_line_iterator(self, url: str):
        for l in read_lines_from_zst(url):
            yield self.parse_with_sep(l)

    def line_iterator(self, url: str):
        if url.endswith(".zst"):
            return self.zst_line_iterator(url)
        else:
            return self.gzip_line_iterator(url)

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        raise NotImplementedError()

    def get_n_shards(self, split: Optional[str] = None) -> int:
        raise NotImplementedError()

    def get_tokenizer_n_files(self):
        return self.TOKENIZER_N_FILES

    def get_tokenizer_train_sentences(self):
        n_files = min(self.get_tokenizer_n_files(), self.get_n_shards("train"))
        for i in range(n_files):
            url = self.get_url(i, "train")
            for txt in self.line_iterator(url):
                yield txt

    def _chunk_fname(self, index: int) -> str:
        return os.path.join(self._chunk_dir, f"chunk_{index}.bin")

    def tokenize_and_write(self, ofile, batch: List[str]):
        # Some huggingface tokenizers are fast only in batches.
        tokenized = self.vocabulary.tokenize_batch(batch)
        for l in tokenized:
            np.asarray(l, dtype=self.data_dtype).tofile(ofile)

    def tokenize_chunk(self, chunk_index):
        fname = self._chunk_fname(chunk_index)
        if not os.path.exists(fname):
            print(f"Tokenizing chunk {chunk_index}...")

            url = self.get_url(chunk_index)
            lines = []

            with open(fname+".tmp", "wb") as out_f:
                try:
                    for l in self.line_iterator(url):
                        lines.append(l)
                        if len(lines) > 1000:
                            self.tokenize_and_write(out_f, lines)
                            lines.clear()
                except:
                    print(f"Exception while tokenizing chunk {chunk_index}, url: {url}")
                    raise

                if lines:
                    self.tokenize_and_write(out_f, lines)

            os.rename(fname+".tmp", fname)
            print(f"Tokenizing chunk {chunk_index} done.")

    def update_chunk_access_time(self, index: int):
        # Prevent too many files and being killed by Linux if the
        # dataset have many chunks
        is_new = self.chunk_mmap[index] is None
        now = time.time()
        self.chunk_access_times[index] = now

        if is_new:
            # If a new file is opened, check if there is an old one to be closed.
            to_close = [k for k, v in self.chunk_access_times.items() if now - v > self.CHUNK_CLOSE_TIMEOUT]
            for k in to_close:
                assert self.chunk_mmap[k] is not None
                old = self.chunk_mmap[k]
                self.chunk_mmap[k] = None

                del old
                del self.chunk_access_times[k]

    def do_mmap(self, index: int):
        self.update_chunk_access_time(index)
        if self.chunk_mmap[index] is None:
            fname = get_cached_file(self._chunk_fname(index))
            self.chunk_mmap[index] = np.memmap(fname, dtype=self.data_dtype, mode='r')

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

    def get_chunk_sizes(self) -> List[int]:
        chunk_sizes = []
        for i in range(self._n_chunks):
            fn = self._chunk_fname(i)
            if os.path.exists(fn):
                chunk_sizes.append(os.path.getsize(fn) // self.data_dtype(0).itemsize)
            else:
                break
        return chunk_sizes

    def get_ready_tokens(self) -> int:
        return sum(self.get_chunk_sizes())

    def get_min_n_chunks(self) -> int:
        return 1

    def prepare(self):
        pass

    def __init__(self, unroll_len: int, n_extra: int = 1, split: str = 'train',
                 cache_dir: str = "./cache/", n_tokens: int = 8000, token_limit: Optional[int] = None,
                 shuffle: bool = False, vocabulary: Optional[Vocabulary] = None) -> None:
        self.split = split
        self.n_tokens = n_tokens if vocabulary is None else len(vocabulary)
        self.unroll_len = unroll_len
        self.n_extra = n_extra
        self.external_vocabulary = vocabulary
        self.update_data_type()

        self._cache_dir_base = os.path.join(cache_dir, self.__class__.__name__)
        self._cache_dir = os.path.join( self._cache_dir_base, self._get_variant_id())
        self._chunk_dir = os.path.join(self._cache_dir, "tokenized_chunks", split)
        self.last_available_chunk = -1
        self.last_accessed_chunk = -1
        self.token_limit = int(math.ceil(token_limit)) if token_limit is not None else None

        os.makedirs(self._chunk_dir, exist_ok=True)

        self.prepare()

        self._n_chunks = self.get_n_shards()
        self.chunk_sizes = [0] * self._n_chunks
        self.chunk_offsets = [0] * self._n_chunks
        self.chunk_mmap = [None] * self._n_chunks
        self.chunk_access_times = {}

        self._sp_model_name = os.path.join(self._cache_dir, "tokenizer.model")

        with LockFile(self._cache_dir + "/lock"):
            if vocabulary is None:
                self.vocabulary = SentencepieceVocabulary(self._sp_model_name, GenToIt(self.get_tokenizer_train_sentences), n_tokens)
                print(f"{self.__class__.__name__}: Loaded tokenizer.")
            else:
                self.vocabulary = vocabulary

            missing = [i for i in range(self._n_chunks) if not os.path.exists(self._chunk_fname(i))]
            print(f"{self.__class__.__name__}: {len(missing)} chunks missing")
            if missing:
                if token_limit is not None:
                    n_proc = min(mp.cpu_count(), len(missing), 64)
                    pool = mp.Pool(n_proc)

                    while True:
                        tokens_ready = self.get_ready_tokens()
                        chunks_ready = len(self.get_chunk_sizes())

                        if tokens_ready >= token_limit:
                            if chunks_ready >= self.get_min_n_chunks():
                                print("Token limit reached. No need to tokenize more.")
                                break

                        print(f"{self.__class__.__name__}: {tokens_ready/token_limit*100:.2f}% ready.")

                        if chunks_ready == 0:
                            print("Tokenizing first chunk to estimate the number of required chunks...")
                            self.tokenize_chunk(0)
                            continue
                        elif chunks_ready >= self._n_chunks:
                            print("All chunks ready. No need to tokenize more.")
                            break

                        n_estimated = max(int(math.ceil(chunks_ready * (token_limit / tokens_ready))), self.get_min_n_chunks())

                        print(f"{self.__class__.__name__}: Tokenizing {n_estimated} estimated chunks...")
                        if self.vocabulary.allow_multiprocessing():
                            pool.map(self.tokenize_chunk, [a for a in range(chunks_ready, n_estimated) if a in missing])
                        else:
                            for a in range(chunks_ready, n_estimated):
                                if a in missing:
                                    self.tokenize_chunk(a)
                    del pool
                else:
                    mp.Pool(min(mp.cpu_count(), len(missing))).map(self.tokenize_chunk, missing)

        self.chunk_sizes = self.get_chunk_sizes()
        self.chunk_offsets = self.chunk_offsets[:len(self.chunk_sizes)]

        lim_found = False
        chunk_limit = len(self.chunk_sizes)
        for i in range(1, len(self.chunk_sizes)):
            self.chunk_offsets[i] = self.chunk_offsets[i - 1] + self.chunk_sizes[i]
            if self.token_limit is not None and not lim_found and self.chunk_offsets[i] >= self.token_limit:
                print(f"{self.__class__.__name__}: Limiting to first {i} chunks because limited to {self.token_limit} tokens")
                lim_found = True
                chunk_limit = i

        if self.token_limit is not None:
            # We need this to ensure that if random indices are used, we are always selecting the same subset
            chunk_limit = max(chunk_limit, self.get_min_n_chunks())
            self.chunk_sizes = self.chunk_sizes[:chunk_limit]
            self.chunk_offsets = self.chunk_offsets[:chunk_limit]

        self.chunk_mmap = self.chunk_mmap[:len(self.chunk_sizes)]

        if shuffle:
            if self.token_limit is not None:
                print(f"{self.__class__.__name__}: WARNING: Shuffling guarantuees indentical data output only if identical token limit and unroll length is used.")

            self.index_remap_order = np.random.default_rng(123).permutation(sum(self.chunk_sizes) // self.unroll_len).tolist()
            self.index_remap = lambda x: self.index_remap_order[x]
        else:
            self.index_remap = lambda x: x

    def __len__(self):
        l = self.linear_len()
        if self.token_limit is not None:
            l = min(l, self.token_limit)

        return l // self.unroll_len

    def linear_len(self):
        return self.chunk_sizes[-1] + self.chunk_offsets[-1]

    def get_linear(self, offset: int, clen: int):
        chunk_index = bisect.bisect(self.chunk_offsets, offset) - 1
        chunk_offset = offset - self.chunk_offsets[chunk_index]

        self.do_mmap(chunk_index)

        if chunk_offset + clen > self.chunk_sizes[chunk_index]:
            # Wrapping over chunk boundary
            next_chunk = (chunk_index + 1) % len(self.chunk_sizes)
            self.do_mmap(next_chunk)

            d1 = self.chunk_mmap[chunk_index][chunk_offset:]
            d2 = self.chunk_mmap[next_chunk][:clen-len(d1)]

            r = np.concatenate([d1, d2])
        else:
            r = self.chunk_mmap[chunk_index][chunk_offset:chunk_offset+clen]

        assert r.shape[0] == clen
        return r

    def __getitem__(self, item: int) -> Dict[str, Any]:
        item = self.index_remap(item)
        return {
            "data": self.get_linear(item * self.unroll_len, self.unroll_len + self.n_extra)
        }

    def start_test(self) -> WordLevelLanguageModelTestState:
        return WordLevelLanguageModelTestState()

