# Replication
In this repository you can find code to replicate our examples and benchmark results.
Contents:
- evaluation - replicates the benchmarks of our LLM models.
- translation-API - replicates the example usages of our API. (SOON)


# Evaluating Translation Models 

This code can be used to replicate the benchmarks on the en-nl (https://huggingface.co/Michielo/mt5-small_en-nl_translation) and nl-en (https://huggingface.co/Michielo/mt5-small_nl-en_translation) translation models. Evalution is done on BLEU, chrF, and chrF++ metrics.

### Installation:

- Clone the repository using ``git clone https://github.com/AssistantsLab/Replication.git``
- Navigate to the project directory and run the command ``pip install -r requirements.txt``
### Code Structure:

- `main.py`: The main script that parses command-line arguments and runs the evaluation.
- `evaluate.py`: Contains the `Evaluate` class with methods for loading the model, generating hypotheses, and computing evaluation metrics.
- `data/dev.txt`: File containing source and target texts for evaluation
### Using the program 

- Run the `main.py` script with the desired translation model type:
- ```python main.py --model <model_type>```

- Replace `<model_type>` with either `en-nl` for evaluating English-to-Dutch translation model or `nl-en` for evaluating Dutch-to-English translation model
Example: ```python main.py --model en-nl```
