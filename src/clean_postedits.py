#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import json


def process_level(input_file, output_dir, level):
    # Read all sheets from the Excel file
    sheets_dict = pd.read_excel(input_file, engine="openpyxl", sheet_name=None)

    # Dictionary mapping for level-based configuration
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

    # Use the mapping for the chosen level
    selected = mapping[level]
    keys = [
        key
        for key in list(sheets_dict.keys())
        if key.split()[-1].lower() == selected["suffix"]
    ]
    level_output_dir = os.path.join(output_dir, f"{selected['subdir']}")
    json_output_file = os.path.join(output_dir, f"{selected['json']}")

    # Create the output directory if it doesn't exist
    os.makedirs(level_output_dir, exist_ok=True)

    langs = []
    for lang in keys:
        df = sheets_dict[lang]
        # Mark rows where "Original" contains "Translation" or is empty
        df["flag"] = df["Original"].apply(
            lambda x: (
                (("Translation" in x.split()) or (len(x.strip().split()) == 0))
                if isinstance(x, str)
                else False
            )
        )
        data = pd.DataFrame()
        data["Generated"] = df["Original"].loc[~df["flag"]]
        data["Corrected"] = df["MNSW_post-edited"].loc[~df["flag"]]
        data = data.dropna().reset_index(drop=True)

        # Derive language name from the sheet name and format file path
        lang_name = lang.replace("_", " ").split()[0].lower()
        output_path = os.path.join(level_output_dir, f"{lang_name}.csv")
        data.to_csv(output_path, index=False)
        langs.append(lang_name)

    # Write out the list of languages as a JSON file
    with open(json_output_file, "w") as json_file:
        json.dump(langs, json_file, indent=4)

    print(f"Processed {len(langs)} languages for level '{level}'.")
    print(f"CSV files are saved in: {level_output_dir}")
    print(f"JSON file written to: {json_output_file}")


def main():
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
