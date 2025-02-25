#!/usr/bin/env python3
"""
Process NSWEduChat Translations
--------------------------------
This script processes NSWEduChat translations stored in an Excel file.
It extracts translation data from each sheet and creates individual text files
for each language as well as a combined CSV file containing all translations.

Usage:
    python process_translations.py --input_file <path_to_excel> --output_dir <output_directory>

Arguments:
    --input_file  : Path to the input Excel file containing NSWEduChat translations.
    --output_dir  : Directory where the processed text files and CSV file will be saved.

Dependencies:
    - argparse
    - os
    - pandas (with openpyxl engine for reading Excel files)
"""

import argparse
import os
import pandas as pd


def process_translations(input_file, output_dir):
    """
    Process translations from the provided Excel file.

    This function reads all sheets from the Excel file, extracts the "English paragraph"
    from the "Arabic" sheet as the reference English translations, and then extracts the
    "NSWEduChat translation" from every sheet. It writes each language's translations to
    individual text files and also saves a combined CSV file containing all translations.

    Args:
        input_file (str): Path to the input Excel file.
        output_dir (str): Directory to save the processed text files and CSV file.
    """
    # Read all sheets from the Excel file using the openpyxl engine
    sheets_dict = pd.read_excel(input_file, engine="openpyxl", sheet_name=None)

    # Create a DataFrame to hold the translations
    data = pd.DataFrame()

    # Use the "Arabic" sheet's "English paragraph" as the english column
    data["english"] = sheets_dict["Arabic"]["English paragraph"]

    # For each sheet in the Excel file, add a column for the corresponding language
    # using the "NSWEduChat translation" field.
    for lang in sheets_dict.keys():
        data[lang.lower()] = sheets_dict[lang]["NSWEduChat translation"]

    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Write each column (language) to a separate text file in the output directory.
    for lang in data.columns:
        output_txt_path = os.path.join(output_dir, f"{lang.lower()}.txt")
        data[lang].to_csv(output_txt_path, index=False, header=False)

    # Write the full DataFrame with all translations to a CSV file.
    output_csv_path = os.path.join(output_dir, "all_translations.csv")
    data.to_csv(output_csv_path, index=False)

    print(
        f"Processed translations and saved individual text files and CSV file to '{output_dir}'."
    )


def main():
    """
    Main function to parse command-line arguments and process translations.

    This function sets up the argument parser, retrieves the input file path and output
    directory from the command-line arguments, and calls the process_translations function.
    """
    parser = argparse.ArgumentParser(
        description="Process NSWEduChat translations from an Excel file."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input Excel file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the text and CSV files.",
    )
    args = parser.parse_args()
    process_translations(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
