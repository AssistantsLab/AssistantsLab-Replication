import sacrebleu


def compute_chrf_plus_plus(hypotheses, references):
    """
    Compute the chrF++ score
    Args:
        hypotheses (list): List of hypotheses (translations)
        references (list): List of reference texts (self.trgt_text)
    Returns:
        float: The chrF++ score
    """
    chrf_plus_plus = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2, char_order=6, beta=2,
                                           eps_smoothing=False).score
    return chrf_plus_plus
