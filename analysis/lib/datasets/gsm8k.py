import datasets

class GSM8K:
    def __init__(self):
        self.dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")

    @staticmethod
    def format_example(example):
        question = example["question"]
        answer = example["answer"].split("####")
        assert len(answer) == 2
        res = f"Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"
        return f"{res}\n{answer[0]}The final answer is {answer[1].strip()}"
    
    def __iter__(self):
        for example in self.dataset:
            yield self.format_example(example)
