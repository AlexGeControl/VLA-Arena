import argparse
from .train import train_main
from .eval import eval_main

def main():
    parser = argparse.ArgumentParser("vla-arena CLI")
    sub = parser.add_subparsers(dest="cmd")

    # train
    train_p = sub.add_parser("train")
    train_p.add_argument("--model", required=True)
    train_p.add_argument("--config", default=None)
    train_p.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint directory")

    # eval
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--config", default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "eval":
        eval_main(args)
    else:
        parser.print_help()
