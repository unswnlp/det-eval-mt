#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def process_translations(input_file, output_dir):
    # Read all sheets from the Excel file
    sheets_dict = pd.read_excel(input_file, engine="openpyxl", sheet_name=None)

    # Create a DataFrame to hold the translations
    data = pd.DataFrame()

    # Use the "Arabic" sheet's "English paragraph" as the english column
    data["english"] = sheets_dict["Arabic"]["English paragraph"]

    # For each sheet in the Excel file, add a column using the "NSWEduChat translation"
    for lang in sheets_dict.keys():
        data[lang.lower()] = sheets_dict[lang]["NSWEduChat translation"]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write each column to a separate text file in the output directory
    for lang in data.columns:
        output_txt_path = os.path.join(output_dir, f"{lang.lower()}.txt")
        data[lang].to_csv(output_txt_path, index=False, header=False)

    # Write the full DataFrame to a CSV file in the output directory
    output_csv_path = os.path.join(output_dir, "all_translations.csv")
    data.to_csv(output_csv_path, index=False)

    print(
        f"Processed translations and saved individual text files and CSV file to '{output_dir}'."
    )


def main():
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
