import re
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

class Evaluate:
        
    def __init__(self, type):
        """
        Initialize the Evaluate class

        Args:
            type (str): The type of translation model to be evaluated ("en-nl" or "nl-en")
        """        
        self.model_type = type
        self.model_name = None
        self.prefix = None
        self.src_text = []
        self.trgt_text = []
        self.hypotheses = []
        self.file_path = "data/dev.txt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_config()

    def set_config(self):
        """
        Set the model name and prefix based on the specified model type
        """    
        if self.model_type == "en-nl":
            self.model_name = "Michielo/mt5-small_en-nl_translation"
            self.prefix = ">>nl<<"
        elif self.model_type == "nl-en":
            self.model_name = "Michielo/mt5-small_nl-en_translation"
            self.prefix = ">>en<<"

    def load_model(self):
        """
        Load the tokenizer and model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def read_file(self):
        """
        Read and set the source and target texts from the specified file based on the model type
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            eng_text = []
            nl_text = []
            for line in lines:
                parts = re.split(r'\t+', line.strip())
                if len(parts) == 4:
                    eng_text.append(parts[2])
                    nl_text.append(parts[3])
            if self.model_type == "en-nl":
                self.src_text = eng_text
                self.trgt_text = nl_text
            elif self.model_type == "nl-en":
                self.src_text = nl_text
                self.trgt_text = eng_text

    def prepare_data(self):
        """
        Prepare the source and target texts, add appropriate prefix to the self.src_text
        """
        self.read_file()
        self.src_text = [f"{self.prefix} {text}" for text in self.src_text]   

    def generate_hypotheses(self):
        """
        Generate hypotheses (translations) for the source texts using the loaded model
        """
        for src_text in tqdm(self.src_text, desc="Generating hypotheses"):
            inputs = self.tokenizer(src_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.hypotheses.extend(decoded_output)

    def compute_bleu(self, hypotheses, references):
        """
        Compute the BLEU score
        Args:
            hypotheses (list): List of hypotheses (translations)
            references (list): List of reference texts (self.trgt_text)
        Returns:
            float: The BLEU score
        """
        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        return bleu

    def compute_chrf(self, hypotheses, references):
        """
        Compute the chrF score
        Args:
            hypotheses (list): List of hypotheses (translations)
            references (list): List of reference texts (self.trgt_text)
        Returns:
            float: The chrF score
        """
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
        return chrf

    def compute_chrf_plus_plus(self, hypotheses, references):
        """
        Compute the chrF++ score
        Args:
            hypotheses (list): List of hypotheses (translations)
            references (list): List of reference texts (self.trgt_text)
        Returns:
            float: The chrF++ score
        """
        chrf_plus_plus = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2, char_order=6, beta=2, eps_smoothing=False).score
        return chrf_plus_plus

    def compute_scores(self):
        """
        Compute the BLEU, chrF, and chrF++ scores for the generated hypotheses
        """
        bleu_score = self.compute_bleu(self.hypotheses, self.trgt_text)
        print(f"BLEU score: {bleu_score:.4f}")
        
        chrf_score = self.compute_chrf(self.hypotheses, self.trgt_text)
        print(f"chrF score: {chrf_score:.4f}")

        chrf_plus_plus_score = self.compute_chrf_plus_plus(self.hypotheses, self.trgt_text)
        print(f"chrF++ score: {chrf_plus_plus_score:.4f}")

    def evaluate_model(self):
        """
        Evaluate the model by preparing data, generating hypotheses and computing scores
        """
        self.load_model()
        self.prepare_data()
        self.generate_hypotheses()
        self.compute_scores()


