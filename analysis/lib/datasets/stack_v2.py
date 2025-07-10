import datasets

class STACK_V2:
    def __init__(self):
        self.dataset = datasets.load_dataset("stalaei/stack-v2-subset-all-above-15k-tokens", split="train")

    @staticmethod
    def format_example(example):
        blob_id = example["blob_id"]
        code = example["text"]
        language = example["language"]

        res = f"Read the following code and understand it carefully.\n'''{language}\n{code}\n'''"
        return res
    
    def __iter__(self):
        for example in self.dataset:
            yield self.format_example(example)
