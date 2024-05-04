import sacrebleu


def compute_chrf(hypotheses, references):
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