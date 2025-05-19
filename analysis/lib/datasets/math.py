import datasets
import random


class Math:
    def __init__(self, level: int, subset: str = "intermediate_algebra"):
        self.dataset = []
        for d in datasets.load_dataset("EleutherAI/hendrycks_math", subset, split="test"):
            if int(d["level"][-1]) == level:
                self.dataset.append(d)

        if len(self.dataset) == 0:
            raise ValueError(f"No dataset found for level {level}")
        
        rng = random.Random(42)
        rng.shuffle(self.dataset)

    @staticmethod
    def levels():
        return list(range(1, 6))
    
    def format_example(self, example):
        return f"Problem: {example['problem']}\nAnswer:", f" {example['solution']}"
    
    @staticmethod
    def level_format():
        return "Level {}"
    
    def __iter__(self):
        for example in self.dataset:
            yield self.format_example(example)
