import datasets
import random

class MQuake:
    def __init__(self, n_hops=2):
        self.data = []

        for d in datasets.load_dataset("henryzhongsc/MQuAKE-Remastered", split="CF3k"):
            if len(d["orig_triples"]) == n_hops:
                self.data.append(d)

        rng = random.Random(42)
        rng.shuffle(self.data)

    @staticmethod
    def levels():
        return [2, 3, 4]

    @staticmethod
    def format_example(d):
        question = d["questions"][0]
        answer = d["answer"]
        return f"Question: {question}\nAnswer:", f" {answer}"
    
    @staticmethod
    def level_format():
        return "{} hops"
    
    def __iter__(self):
        for example in self.data:
            yield self.format_example(example)
