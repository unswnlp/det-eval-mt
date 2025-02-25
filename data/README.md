## Data Directory

This repository contains translation data from the NSWEduChat project. The data is organized into raw files and processed directories for both Educhat translations and post-edited translation pairs. Below is an overview of the files and directories:

### Raw Data Files

`NSWEduChat_all languages_60 translations[100].xlsx`: Contains Educhat translations in multiple languages. Each sheet represents translations for a specific language with a total of 60 translations (from a possible 100).
`Languages_by_sentence and paragraph.xlsx`:  Contains post-edited translation pairs. This file includes both sentence-level and paragraph-level data, where the original translations have been revised (post-edited).

### Processed Educhat Translations

All processed Educhat translations are stored in the educhat-translation directory. 
- `educhat-translation/all_translations.csv`: A combined CSV file that aggregates translations from all languages, providing an easy-to-access format for further analysis.
- `educhat-translation/{language}.txt`: Individual text files for each language containing the translations. Replace {language} with the respective language name in lowercase (e.g., english.txt, arabic.txt, etc.).
- `educhat-translation/types.csv`: A CSV file detailing the use cases for the Educhat translations.

### Post-Edited Translation Pairs

Post-edited translations (i.e., the corrected versions) have been processed at two levels: paragraph and sentence. These are organized in the post-edited directory.

- `post-edited/paragraph/{language}.csv`: CSV files containing paragraph-level post-edited translations for each language.
**Note:** The Korean post-edited translations are missing.
- paragraph.json : A JSON file listing all languages for which paragraph-level post-edited translations are available.
**Note:** Korean is not included in this list.
- `post-edited/sentence/{language}.csv`: CSV files containing sentence-level post-edited translations for each language.
**Note:** The Tamil post-edited translations are missing.
- `sentence.json`: A JSON file listing all languages for which sentence-level post-edited translations are available.
**Note:** Tamil is not included in this list.

## Directory Structure Overview
```
.
├── NSWEduChat_all languages_60 translations[100].xlsx   # Educhat translations (raw data)
├── Languages_by_sentence and paragraph.xlsx             # Post-edited translation pairs (raw data)
├── educhat-translation/
│   ├── all_translations.csv      # Combined processed translations
│   ├── types.csv                 # Use cases for Educhat translations
│   ├── english.txt               # English translations (example)
│   ├── arabic.txt                # Arabic translations (example)
│   └── {language}.txt            # Other language translation text files
└── post-edited/
    ├── paragraph/
    │   └── {language}.csv        # Paragraph-level post-edited translations (Korean missing)
    └── sentence/
        └── {language}.csv        # Sentence-level post-edited translations (Tamil missing)
├── paragrapgh.json               # JSON list of languages with paragraph-level post-edited translations (Korean missing)
└── sentence.json                 # JSON list of languages with sentence-level post-edited translations (Tamil missing)
```
**Notes**
- Replace {language} with the specific language name (e.g., english, arabic, bengali, etc.).
- The processed Educhat translations (all_translations.csv and individual {language}.txt files) are created for easier accessibility and analysis.
- Post-edited translations have been separated into sentence-level and paragraph-level CSV files.
- Missing Data:
    - Korean is missing in the paragraph-level post-edited translations.
	- Tamil is missing in the sentence-level post-edited translations.
- The JSON files (paragraph.json and sentence.json) provide a quick reference to the languages available in each post-edited category.

