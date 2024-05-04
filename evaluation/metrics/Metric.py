from enum import Enum


class Metric(Enum):
    """
    This enum holds all possible metrics to be computed by this translation evaluation suite.
    This is purely to avoid confusion.
    """
    BLEU = "bleu"
    CHRF = "chrf"
    CHRF_PLUS_PLUS = "chrf_plus_plus"