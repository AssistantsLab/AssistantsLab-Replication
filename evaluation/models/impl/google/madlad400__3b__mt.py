import re

from tqdm import tqdm

from models.model import ModelClass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Model(ModelClass):

    def __init__(self, name):
        super().__init__(name)
        self.src_text = []
        self.trgt_text = []
        self.hypotheses = []

    def load_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name, device_map="auto", offload_folder=("offload_" + self.name))
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    def prep(self):
        file_path = "data/en-fr/dev.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            eng_text = []
            fr_text = []
            for line in lines:
                parts = re.split(r'\t+', line.strip())
                if len(parts) == 4:
                    eng_text.append(parts[2])
                    fr_text.append(parts[3])
                self.src_text = eng_text
                self.trgt_text = fr_text
        self.src_text = [f"<2fr> {text}" for text in self.src_text]

    def run_inference(self):
        for src_text in tqdm(self.src_text, desc="Generating hypotheses"):
            inputs = self.tokenizer(src_text, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs)
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.hypotheses.extend(decoded_output)
        return self.hypotheses

    def get_references(self):
        return self.trgt_text
