#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import evaluate


def compute_scores(csv_dir, languages):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    chrf = evaluate.load("chrf")

    entries = []
    for lang in tqdm(languages, desc="Processing languages"):
        csv_file = os.path.join(csv_dir, f"{lang}.csv")
        try:
            data = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        try:
            bleu_score = bleu.compute(
                predictions=list(data["Generated"]),
                references=[[i] for i in data["Corrected"]],
            )
            rouge_score = rouge.compute(
                predictions=list(data["Generated"]),
                references=[[i] for i in data["Corrected"]],
            )
            chrf_score = chrf.compute(
                predictions=list(data["Generated"]),
                references=[[i] for i in data["Corrected"]],
            )

            entry = {
                "language": lang.replace("_", " ").split()[0].lower(),
                "bleu": bleu_score,
                "rouge": rouge_score,
                "chrf": chrf_score,
                "mean": (
                    chrf_score["score"] / 100
                    + bleu_score["bleu"]
                    + rouge_score["rouge1"]
                    + rouge_score["rouge2"]
                    + rouge_score["rougeL"]
                )
                / 5,
            }
            entries.append(entry)
        except Exception as e:
            print(f"Error processing {lang} from {csv_file}: {e}")
            continue
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for translations at sentence or paragraph level."
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["sentence", "paragraph"],
        help="Translation level to process: 'sentence' or 'paragraph'.",
    )
    args = parser.parse_args()
    level = args.level.lower()

    # Build OS-agnostic file paths using os.path.join
    input_json = os.path.join("data", "post-edited", f"{level}.json")
    csv_dir = os.path.join("data", "post-edited", level)
    output_json = os.path.join("results", f"{level}.json")

    try:
        with open(input_json, "r") as f:
            languages = json.load(f)
    except Exception as e:
        print(f"Error reading {input_json}: {e}")
        return

    entries = compute_scores(csv_dir, languages)

    try:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        pd.DataFrame(entries).to_json(output_json, orient="records", indent=4)
        print(f"Results written to {output_json}")
    except Exception as e:
        print(f"Error writing to {output_json}: {e}")


if __name__ == "__main__":
    main()
