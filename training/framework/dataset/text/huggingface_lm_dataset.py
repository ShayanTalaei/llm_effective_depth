import multiprocessing
import os
import numpy as np
from tqdm import tqdm
import json
import queue
from ..sequence_dataset import SequenceDataset
from .tokenizers.huggingface import HuggingfaceVocabulary
from .lm_dataset import WordLevelLanguageModelTestState
from ..fs_cache import get_cached_file
import torch.multiprocessing as mp
from typing import Any
import time
from ...utils import LockFile


datasets = None
transformers = None

def load_huggingface():
    global datasets
    global transformers

    import datasets
    import transformers



class AsyncStreamingLoader:
    def __init__(
        self,
        dataset_name: str,
        shuffle_block_size: int,
        queue_chunk_size: int,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle_block_size = shuffle_block_size
        self.queue_chunk_size = queue_chunk_size

        # Initialize multiprocessing queue and event
        self.queue = mp.Queue(maxsize=2)
        self.stop_event = mp.Event()

    def _load_and_queue_data(self) -> None:
        seed = np.random.RandomState(42)

        dataset = datasets.load_dataset(
            self.dataset_name,
            streaming=True,
            split=self.split,
            trust_remote_code=True,
            # buffer_size=2*self.block_size
        )

        dataset = dataset.shuffle(buffer_size=self.shuffle_block_size)

        current_batch = []

        items = None
        for items in dataset.batch(self.queue_chunk_size):
            if self.stop_event.is_set():
                break

            while not self.stop_event.is_set():
                try:
                    self.queue.put(items["text"], timeout=0.1)
                    break
                except queue.Full:
                    time.sleep(0.1)

        # items = None
        # while not self.stop_event.is_set():
        #     if items is None:
        #         items = dataset.take(self.block_size)

        #     print("TAKE RETURNED LENGTH", len(items))
        #     try:
        #         self.queue.put(items, timeout=0.1)
        #         items = None
        #     except queue.Full:
        #         time.sleep(0.1)

        # for item in dataset:
        #     if self.stop_event.is_set():
        #         break

        #     try:
        #         if len(current_batch) == self.block_size:
        #             seed.shuffle(current_batch)
        #             self.queue.put(current_batch, timeout=0.1)
        #             current_batch = []
        #         else:
        #             current_batch.append(item["text"])
        #     except queue.Full:
        #         time.sleep(0.1)

        self.queue.put(None)

    def start(self) -> None:
        self.process = mp.Process(target=self._load_and_queue_data)
        self.process.start()

    def stop(self) -> None:
        self.stop_event.set()
        while True:
            if self.queue.get() is None:
                break
        self.process.join()



class HuggingfaceLMDataset(SequenceDataset):
    VERSION = 1

    def my_id(self) -> str:
        return f"{self.dataset}_{self.tokenizer.name_or_path}_{self.n_tokens}_{self.split}_{self.randomize_chunk_size}"

    def __init__(self, dataset: str, tokenizer, context_length: int, n_tokens: int, split: str = "train",
                 cache_path: str = "./cache", randomize_chunk_size: int = 10000):
        self.dataset = dataset
        self.context_length = context_length
        self.split = split
        self.n_tokens = n_tokens
        self.cache_path = cache_path
        self.randomize_chunk_size = randomize_chunk_size

        load_huggingface()
        self.tokenizer = tokenizer
        self.vocabulary = HuggingfaceVocabulary(self.tokenizer)
        self.in_vocabulary = self.vocabulary
        self.out_vocabulary =  self.vocabulary

        self.my_path = os.path.join(self.cache_path, self.__class__.__name__, self.my_id())
        os.makedirs(self.my_path, exist_ok=True)

        if self.get_n_storage_tokens() < 2 ** 16:
            self.dtype = np.uint16
        elif self.get_n_storage_tokens() < 2 ** 32:
            self.dtype = np.uint32
        else:
            raise ValueError("Tokenizer vocabulary is too large")

        self.load()

    def get_n_storage_tokens(self) -> int:
        return len(self.tokenizer)

    def load(self):
        if not self.check_if_ok():
            print(f"{self.__class__.__name__}: Tokenized dataset not found. Tokenizing...")
            with LockFile(self.my_path + "/lock.lock"):
                self.prepare()

        fname = get_cached_file(self.get_data_fname())
        self.data = np.memmap(fname, dtype=self.dtype, mode="r")

    def raw_to_output(self, data):
        return {"data": data}

    def __len__(self):
        return (len(self.data)-1) // self.context_length

    def __getitem__(self, idx):
        data = self.data[idx * self.context_length : (idx + 1) * self.context_length + 1].astype(np.int32)
        return self.raw_to_output(data)

    def get_config_fname(self) -> str:
        return f"{self.my_path}/{self.split}_config.json"

    def get_data_fname(self) -> str:
        return f"{self.my_path}/{self.split}.bin"

    def check_if_ok(self):
        config_file = self.get_config_fname()
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                if config["version"] == self.VERSION:
                    return True
        return False

    def process_batch(self, examples):
        ids = self.tokenizer(
            examples,
            truncation=False,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        for e in ids:
            e.append(self.tokenizer.eos_token_id)
        return ids

    def prepare(self):
        block_size = 1024
        loader = AsyncStreamingLoader(self.dataset, self.randomize_chunk_size, block_size*4, self.split)
        loader.start()
        q = []

        def get_block(size: int):
            nonlocal q
            nonlocal loader
            while len(q) < size:
                s = time.time()
                newblock = loader.queue.get()
                newblock.extend(q)
                q = newblock
                d = time.time() - s
                if d>1:
                    print(f"WARNING: Waiting for data for {time.time() - s}s")

            block = q[-size:]
            q = q[:-size]

            return block

        count = 0
        filename = self.get_data_fname()


        with open(filename, "wb") as out_f:
            with tqdm(total=self.n_tokens, desc=f'tokenizing') as pbar:
                while count < self.n_tokens:
                    data = get_block(block_size)
                    data = self.process_batch(data)
                    for d in data:
                        remaining = self.n_tokens - count
                        if remaining <= 0:
                            break

                        arr = np.array(d, dtype=self.dtype)
                        arr = arr[:remaining]
                        count += len(arr)
                        arr.tofile(out_f)

                        pbar.update(len(arr))

            out_f.flush()


            with open(self.get_config_fname(), "w") as f:
                f.write(json.dumps({"version": self.VERSION}))

        loader.stop()

    def start_test(self) -> WordLevelLanguageModelTestState:
        return WordLevelLanguageModelTestState(ignore_index=self.tokenizer.pad_token_id)