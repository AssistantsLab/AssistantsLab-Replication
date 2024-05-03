from metrics.Metric import Metric
from metrics.bleu import compute_bleu
from metrics.chrf import compute_chrf
from metrics.chrf_plus_plus import compute_chrf_plus_plus

class ModelClass:

    def __init__(self, name):
        self.name = name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        pass

    def prep(self):
        pass

    def run_eval(self, metrics):
        results = self.run_inference()
        references = self.get_references()

        metric_results = {}

        if Metric.BLEU in metrics:
            metric_results[Metric.BLEU] = compute_bleu(
                hypotheses=results,
                references=references
            )
        if Metric.CHRF in metrics:
            metric_results[Metric.CHRF] = compute_chrf(
                hypotheses=results,
                references=references
            )
        if Metric.CHRF_PLUS_PLUS in metrics:
            metric_results[Metric.CHRF_PLUS_PLUS] = compute_chrf_plus_plus(
                hypotheses=results,
                references=references
            )

        for key, value in metric_results.items():
            print(f"{key.name} score: {value:.4f}")
        print("Finished computing!")

    def run_inference(self):
        pass

    def get_references(self):
        pass
