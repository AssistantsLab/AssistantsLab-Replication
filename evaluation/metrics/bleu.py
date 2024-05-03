import sacrebleu


def compute_bleu(hypotheses, references):
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
