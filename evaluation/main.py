import argparse
from evaluate import Evaluate


def main():
    
    parser = argparse.ArgumentParser(description="Select the type of translation model.")

    parser.add_argument("--model", choices=["en-nl", "nl-en"], required=True,
                        help="The type of translation model (en-nl or nl-en).")
    
    args = parser.parse_args()

    print(f"Model '{args.model}' selected.")

    eval = Evaluate(args.model)
    eval.evaluate_model()
    

if __name__ == "__main__":
    main()