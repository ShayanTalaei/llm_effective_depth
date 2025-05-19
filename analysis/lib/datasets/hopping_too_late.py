import os
import csv
from ..download import download

URL = "https://raw.githubusercontent.com/edenbiran/HoppingTooLate/refs/heads/main/datasets/two_hop.csv"


class HoppingTooLate():
    def __init__(self):
        self.cache_dir = "cache/hopping_too_late/"

        fname = os.path.basename(URL)
        self.src_name = os.path.join(self.cache_dir, fname)

        if not os.path.exists(self.src_name):
            print(f"Downloading {URL} to {self.src_name}")
            os.makedirs(self.cache_dir, exist_ok=True)
            download(URL, self.cache_dir)
            print("Done.")

        self.data = []

        with open(self.src_name) as csvfile:
            reader = csv.reader(csvfile)
            for lid, line in enumerate(reader):
                line = [x.strip() for x in line]
                if lid == 0:
                    self.header = line
                else:
                    self.data.append(line)

        self.prompt_id = self.header.index("source_prompt")
        self.response_id = self.header.index("e3_label")

    def format_data(self, data):
        prompt = data[self.prompt_id]
        response = data[self.response_id]

        prompt = prompt[:1].upper() + prompt[1:]
        response = " " + response

        return prompt, response
    
    def __iter__(self):
        for data in self.data:
            yield self.format_data(data)

