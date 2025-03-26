#!/usr/bin/env python3
"""
Translation Data Processing Script
------------------------------------
This script processes translation data at either the sentence or paragraph level.
It reads an Excel file containing translation data across multiple sheets (each sheet
corresponding to a language) and filters out rows where the "Original" text either
contains the word "Translation" or is empty. For each language, it extracts the
"Generated" and "Corrected" translations and writes them to a CSV file. It also
writes a JSON file listing the languages that were processed.

Usage:
    python script_name.py --input_file <path_to_excel> --output_dir <output_directory> --level <sentence|paragraph>

Arguments:
    --input_file  : Path to the input Excel file containing translation data.
    --output_dir  : Directory where the processed CSV and JSON files will be saved.
    --level       : Level of translations to process ("sentence" or "paragraph").

Output:
    - CSV files for each processed language will be saved in:
          <output_dir>/<level>/
    - A JSON file listing processed languages will be saved as:
          <output_dir>/<level>.json

Dependencies:
    - argparse
    - os
    - pandas
    - json
    - openpyxl (for reading Excel files)
"""

import argparse
import os
import pandas as pd
import json


def process_level(input_file, output_dir, level):
    """
    Process translation data at a specified level (sentence or paragraph) and generate CSV and JSON outputs.

    The function performs the following steps:
      1. Reads all sheets from the input Excel file.
      2. Filters the sheets to only include those corresponding to the chosen level.
      3. For each sheet (language):
         - Marks rows where "Original" contains the word "Translation" or is empty.
         - Extracts valid rows into a new DataFrame with columns "Generated" and "Corrected".
         - Writes the DataFrame to a CSV file in the level-specific output directory.
         - Collects the language name.
      4. Writes a JSON file listing the processed language names.

    Args:
        input_file (str): Path to the input Excel file.
        output_dir (str): Directory where the processed CSV and JSON files will be saved.
        level (str): Translation level to process, either "sentence" or "paragraph".

    Returns:
        None.
    """
    # Read all sheets from the Excel file using the openpyxl engine.
    sheets_dict = pd.read_excel(input_file, engine="openpyxl", sheet_name=None)

    # Mapping for level-based configuration:
    # - 'suffix': Expected last word in the sheet name.
    # - 'subdir': Output subdirectory for CSV files.
    # - 'json'  : Output JSON filename.
    mapping = {
        "sentence": {
            "suffix": "sentence",
            "subdir": "sentence",
            "json": "sentence.json",
        },
        "paragraph": {
            "suffix": "paragraph",
            "subdir": "paragraph",
            "json": "paragraph.json",
        },
    }

    # Retrieve the configuration for the chosen level.
    selected = mapping[level]
    # Filter sheets: Only include those whose last word matches the configured suffix.
    keys = [
        key
        for key in list(sheets_dict.keys())
        if key.split()[-1].lower() == selected["suffix"]
    ]
    # Define the output directory for CSV files and the JSON output file.
    level_output_dir = os.path.join(output_dir, f"{selected['subdir']}")
    json_output_file = os.path.join(output_dir, f"{selected['json']}")

    # Create the output directory if it doesn't exist.
    os.makedirs(level_output_dir, exist_ok=True)

    langs = []  # List to collect names of processed languages.
    for lang in keys:
        df = sheets_dict[lang]
        # Mark rows where the "Original" column contains "Translation" or is empty.
        df["flag"] = df["Original"].apply(
            lambda x: (
                (("Translation" in x.split()) or (len(x.strip().split()) == 0))
                if isinstance(x, str)
                else False
            )
        )
        # Create a new DataFrame with only the valid rows.
        data = pd.DataFrame()
        data["Generated"] = df["Original"].loc[~df["flag"]]
        data["Corrected"] = df["MNSW_post-edited"].loc[~df["flag"]]
        data = data.dropna().reset_index(drop=True)

        # Derive the language name from the sheet name:
        # Replace underscores with spaces, take the first word, and convert to lowercase.
        lang_name = lang.replace("_", " ").split()[0].lower()
        output_path = os.path.join(level_output_dir, f"{lang_name}.csv")
        data.to_csv(output_path, index=False)
        langs.append(lang_name)

    # Write out the list of processed languages as a JSON file.
    with open(json_output_file, "w") as json_file:
        json.dump(langs, json_file, indent=4)

    # Print summary information.
    print(f"Processed {len(langs)} languages for level '{level}'.")
    print(f"CSV files are saved in: {level_output_dir}")
    print(f"JSON file written to: {json_output_file}")


def main():
    """
    Main function to parse command-line arguments and process translation data.

    This function parses the command-line arguments to obtain:
      - The input Excel file path.
      - The output directory for saving processed files.
      - The translation level to process ("sentence" or "paragraph").
    It then calls the `process_level` function with these parameters.

    Command-line Arguments:
        --input_file : Path to the input Excel file.
        --output_dir : Directory to save processed CSV and JSON files.
        --level      : Level of translations to process ("sentence" or "paragraph").

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Process translation data at sentence or paragraph level."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input Excel file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed CSV and JSON files.",
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["sentence", "paragraph"],
        help="Level of translations to process: 'sentence' or 'paragraph'.",
    )
    args = parser.parse_args()
    process_level(args.input_file, args.output_dir, args.level.lower())


if __name__ == "__main__":
    main()
