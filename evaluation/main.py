import argparse
from metrics.Metric import Metric
import models.impl.Michielo.mt5__small_en__nl_translation
import models.impl.Michielo.mt5__small_nl__en_translation
import models.impl.facebook.nllb__200__3___3B
import models.impl.Helsinki__NLP.opus__mt__en__fr


model_dict = {
    "michielo_en-nl": models.impl.Michielo.mt5__small_en__nl_translation.Model("Michielo/mt5-small_en-nl_translation"),
    "michielo_nl-en": models.impl.Michielo.mt5__small_nl__en_translation.Model("Michielo/mt5-small_nl-en_translation"),
    "nllb_3.3B": models.impl.facebook.nllb__200__3___3B.Model("facebook/nllb-200-3.3B"),
}


def main():
    parser = argparse.ArgumentParser(description="Select the type of translation model and evaluation metrics.")

    parser.add_argument("--model", choices=list(model_dict.keys()), required=True,
                        help="The type of translation model.")

    parser.add_argument("--bleu", action="store_true",
                        help="Include BLEU score in the evaluation.")

    parser.add_argument("--chrf", action="store_true",
                        help="Include chrF score in the evaluation.")

    parser.add_argument("--chrf++", action="store_true", dest="chrf_plus_plus",
                        help="Include chrF++ score in the evaluation.")

    parser.add_argument("--all", action="store_true",
                        help="Include all evaluation metrics (BLEU, chrF, chrF++).")

    args = parser.parse_args()

    metrics = []
    if args.bleu:
        metrics.append(Metric.BLEU)
    if args.chrf:
        metrics.append(Metric.CHRF)
    if args.chrf_plus_plus:
        metrics.append(Metric.CHRF_PLUS_PLUS)
    if not metrics:
        if args.all:
            metrics = [Metric.BLEU, Metric.CHRF, Metric.CHRF_PLUS_PLUS]
        else:
            print("You did not select any evaluation metrics!")
            print("Please run with one or more of the following: '--bleu' '--chrf' '--chrf++' '--all'")
            print("For more help, please run with --help")
            exit(0)

    print(f"Model '{args.model}' selected.")
    if metrics:
        print(f"Evaluation metric(s): {', '.join(metric.name for metric in metrics)}")

    model = model_dict[args.model]

    print('Loading model...')
    model.load_model()
    print('Preparing data...')
    model.prep()
    print('Running evaluation...')
    model.run_eval(metrics)
    

if __name__ == "__main__":
    main()
