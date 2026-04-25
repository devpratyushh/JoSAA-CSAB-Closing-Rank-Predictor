"""
CLI entry point for college predictions.

Usage examples
──────────────
# Train (run once after scraping is done):
  python predict_cli.py train

# Backtest (check model accuracy on held-out year):
  python predict_cli.py backtest

# Predict:
  python predict_cli.py predict \
      --rank 5000 \
      --exam mains \
      --quota AI \
      --seat-type OPEN \
      --gender Gender-Neutral

  python predict_cli.py predict \
      --rank 1200 \
      --exam advanced \
      --quota AI \
      --seat-type OBC-NCL \
      --gender Gender-Neutral \
      --no-reach
"""

import argparse
import sys
import pandas as pd

CSV_PATH = "josaa_ranks.csv"


def cmd_train(args):
    from pipeline.train import train
    train(CSV_PATH)


def cmd_backtest(args):
    from pipeline.evaluate import backtest
    backtest(CSV_PATH, test_year=args.year)


def cmd_predict(args):
    from pipeline.predict import predict, load_model

    # Normalise gender input
    gender = args.gender
    if gender.lower() in ("female", "f", "female-only"):
        gender = "Female-only (including Supernumerary)"
    else:
        gender = "Gender-Neutral"

    model = load_model()
    results = predict(
        rank        = args.rank,
        exam_type   = args.exam.lower(),
        quota       = args.quota.upper(),
        seat_type   = args.seat_type,
        gender      = gender,
        model       = model,
        include_reach = not args.no_reach,
    )

    if results.empty:
        print("No matching colleges found for the given profile.")
        return

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 0)

    round_cols = [c for c in results.columns if c.startswith("R") and c[1:].isdigit()]
    display_cols = ["Institute", "Academic Program Name"] + round_cols + ["Final Pred"]

    for cat in ["safe", "match", "reach"]:
        subset = results[results["Category"] == cat]
        if subset.empty:
            continue
        print(f"\n{'─'*80}")
        print(f"  {cat.upper()} ({len(subset)} options)")
        print(f"{'─'*80}")
        print(subset[display_cols].to_string(index=False))

    print(f"\nTotal: {len(results)} options  "
          f"(rank={args.rank}, exam={args.exam}, quota={args.quota}, "
          f"seat={args.seat_type}, gender={gender})")


def main():
    parser = argparse.ArgumentParser(description="JOSAA College Predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    sub.add_parser("train", help="Train model from josaa_ranks.csv")

    # backtest
    bt = sub.add_parser("backtest", help="Evaluate model accuracy")
    bt.add_argument("--year", type=int, default=None,
                    help="Year to use as test set (default: most recent)")

    # predict
    pr = sub.add_parser("predict", help="Predict colleges for a student")
    pr.add_argument("--rank",      type=int,   required=True,  help="Your rank")
    pr.add_argument("--exam",      type=str,   required=True,
                    choices=["advanced", "mains"],              help="Exam type")
    pr.add_argument("--quota",     type=str,   required=True,
                    help="Quota: AI, HS, OS ...")
    pr.add_argument("--seat-type", type=str,   required=True,
                    help="Seat type: OPEN, OBC-NCL, SC, ST, EWS ...")
    pr.add_argument("--gender",    type=str,   default="Gender-Neutral",
                    help="Gender-Neutral (default) or Female")
    pr.add_argument("--no-reach",  action="store_true",
                    help="Exclude reach colleges")

    args = parser.parse_args()
    {"train": cmd_train, "backtest": cmd_backtest, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
