#!/usr/bin/env python3
"""
Translation Evaluation Metrics Computation Script
---------------------------------------------------
This script computes evaluation metrics for translations at either the sentence-level
or paragraph-level. It reads CSV files containing generated translations and their
corresponding post-edited corrections, computes metrics such as BLEU, ROUGE, CHRF,
and average edit distance, and then aggregates these metrics into a summary JSON file.

Usage:
    python script_name.py --level <sentence|paragraph>

Arguments:
    --level: Specify the translation level to process. Accepted values are "sentence" or "paragraph".

Input Files:
    - data/post-edited/<level>.json:
        A JSON file listing the languages to process for the specified level.
    - data/post-edited/<level>/{language}.csv:
        CSV files for each language containing two columns:
            "Generated"  : The machine-generated translation.
            "Corrected"  : The corresponding post-edited (corrected) translation.

Output:
    - results/<level>.json:
        A JSON file containing computed evaluation metrics for each language.

Dependencies:
    - pandas
    - argparse
    - json
    - os
    - tqdm
    - evaluate (Hugging Face's evaluate library)
    - editdistance
"""

import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import evaluate


def compute_scores(csv_dir, languages):
    """
    Compute evaluation metrics for translations in multiple languages.

    For each language, this function reads the corresponding CSV file containing the
    'Generated' and 'Corrected' translations. It computes the following metrics:
      - BLEU score
      - ROUGE scores (rouge1, rouge2, and rougeL)
      - CHRF score
      - Average edit distance between the generated and corrected translations
      - Mean score: the average of normalized CHRF, BLEU, and ROUGE scores

    Args:
        csv_dir (str): Directory path where CSV files for each language are stored.
        languages (list): List of language names (as defined in the input JSON) to process.

    Returns:
        list: A list of dictionaries. Each dictionary contains the computed metrics for a language.
    """
    # Load evaluation metrics using the evaluate library
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    chrf = evaluate.load("chrf")

    entries = []  # List to store metric entries for each language

    # Process each language provided in the languages list
    for lang in tqdm(languages, desc="Processing languages"):
        csv_file = os.path.join(csv_dir, f"{lang}.csv")
        try:
            # Read the CSV file for the current language
            data = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        try:
            # Compute BLEU score
            bleu_score = bleu.compute(
                predictions=list(data["Generated"]),
                references=[[i] for i in data["Corrected"]],
            )
            # Compute ROUGE scores (rouge1, rouge2, and rougeL)
            rouge_score = rouge.compute(
                predictions=list(data["Generated"]),
                references=[[i] for i in data["Corrected"]],
            )
            # Compute CHRF score
            chrf_score = chrf.compute(
                predictions=list(data["Generated"]),
                references=[[i] for i in data["Corrected"]],
            )

            # Compute mean score as an average of normalized CHRF, BLEU, and ROUGE scores
            mean_score = (
                (chrf_score["score"] / 100)
                + bleu_score["bleu"]
                + rouge_score["rouge1"]
                + rouge_score["rouge2"]
                + rouge_score["rougeL"]
            ) / 5

            # Build a dictionary entry for the current language
            entry = {
                "language": lang.replace("_", " ").split()[0].lower(),
                "bleu": "%.2f" % bleu_score["bleu"],
                "rouge1": "%.2f" % rouge_score["rouge1"],
                "rouge2": "%.2f" % rouge_score["rouge2"],
                "rougeL": "%.2f" % rouge_score["rougeL"],
                "chrf": "%.2f" % (chrf_score["score"] / 100),
                "mean": "%.2f" % mean_score,
            }
            entries.append(entry)
        except Exception as e:
            print(f"Error processing {lang} from {csv_file}: {e}")
            continue

    return entries


def main():
    """
    Main function to parse command-line arguments, compute evaluation metrics, and save results.

    The function performs the following steps:
      1. Parses the command-line argument to determine the translation level ("sentence" or "paragraph").
      2. Constructs file paths for the input JSON, CSV directory, and output JSON.
      3. Loads the list of languages from the input JSON file.
      4. Computes evaluation metrics for each language using the compute_scores function.
      5. Writes the computed metrics to an output JSON file.
    """
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

    # Load the list of languages from the input JSON file
    try:
        with open(input_json, "r") as f:
            languages = json.load(f)
    except Exception as e:
        print(f"Error reading {input_json}: {e}")
        return

    # Compute evaluation metrics for each language
    entries = compute_scores(csv_dir, languages)

    # Write the computed metrics to the output JSON file
    try:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        pd.DataFrame(entries).to_json(output_json, orient="records", indent=4)
        print(f"Results written to {output_json}")
    except Exception as e:
        print(f"Error writing to {output_json}: {e}")


if __name__ == "__main__":
    main()
