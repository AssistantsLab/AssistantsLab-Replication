# Replication
In this repository you can find code to replicate our examples and benchmark results.
Contents:
- evaluation - replicates the benchmarks of our LLM models.
- translation-API - replicates the example usages of our API. (SOON)


# Evaluating Translation Models 

We have made a suite to evaluate translation models and replicate our findings. This is able to be used without writing any code in #Usage, or you can add your own models and metrics in #Developers.

## Usage

This code can be used to replicate the benchmarks on various translation models. Evaluation is done on BLEU, chrF, and chrF++ metrics.

### Installation:

- Clone the repository using ``git clone https://github.com/AssistantsLab/Replication.git``
- Navigate to the project directory and run the command ``pip install -r requirements.txt``

### Code Structure:

- `main.py`: The main script that parses command-line arguments and runs the evaluation.
- `models/`: Contains the models supported in this evaluation suite.
- `metrics/`: Contains the metrics supported in this evaluation suite such as `bleu` and `chrF` among others.
- `data/`: Contains the datasets for various language-pairs such as `nl-en` and `en-nl`.

### Using the program 

Run the `main.py` script with the desired translation model and metrics.

Possible models are:
  - `michielo_en-nl` for the [Michielo/mt5-small_en-nl_translation](https://huggingface.co/Michielo/mt5-small_en-nl_translation) model.
  - `michielo_nl-en` for the [Michielo/mt5-small_nl-en_translation](https://huggingface.co/Michielo/mt5-small_nl-en_translation) model.
  - `nllb_3.3B` for the [facebook/nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) model.

    please note this model takes up ~19GB vram.
  
  Possible metrics are one or more of the following:
  - `bleu`
  - `chrf`
  - `chrf++`
  
    or:
  - `all`

As such, you can run the program as:

```python main.py --model <model_type> [metrics]```

Replace `<model_type>` with one of the supported models and replace `[metrics]` with one or more of the supported metrics.

Examples:

```python main.py --model michielo_en-nl --bleu```

```python main.py --model michielo_en-nl --bleu --chrf```

```python main.py --model michielo_en-nl --all```

```python main.py --model nllb_3.3B --chrf++```


## Developers

We hope that with an easy-to-use code structure you should be able to add support for any translation model and any metric.


### Adding model support
When adding support for a model, please ensure to place this model file in `models/impl/` with the huggingface notation of a model (for example `facebook/nllb__200__3___3B`). Also make sure to put your newly supported model in the `model_dict` that you can find in `main.py`. You can optionally add support to evaluate your newly supported model by adding a parser argument.

We have decided to use the following replacements of special characters in model files. Although these replacements are just our preferences, we believe it would be beneficial to keep this as a standard:
- `-` -> `__`
- `+` -> `___`


### Adding datasets

When adding new datasets, please ensure to keep them in the same format as the existing datasets. This means:
- naming them "dev.txt"
- using a tab as seperator
- formatting as: `<lang_1> \t <lang_2> \t <sentence_1> \t <sentence_2>`
- placing your dataset in the correct language-pair folder