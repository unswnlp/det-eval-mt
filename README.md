## Evaluation of Translation Ability of Language Models for Education-related Communication

**Dipankar Srirag and Dr. Aditya Joshi**

School of Computer Science and Engineering, UNSW Sydney

### Overview

The goal of this project is to support the NSW Department of Education in achieving robust multilingual communication in education using Large Language Models (LLMs). Based on the data provided, we evaluate the performance of language models that translate education-related communications across 22 languages. We employ statistical measures such as chrF, BLEU, and ROUGE to assess translation quality at both sentence and paragraph levels. We also perform reference-less evaluation using multilingual sentence embeddings and identify issues by performing error analysis on post-edited texts, with special focus on Hindi.

### Dataset

The dataset comprises 60 machine-generated translations and 6 post-edited references across five domains (Emails, Circulars, Conversations, Student Essays, and Miscellaneous). An initial analysis revealed missing or incomplete translations for some languages, with limited post-edited references for others.

### Evaluation Metrics

**Reference-based Evaluation**
- chrF (Character-level F-score): Measures similarity using character n-grams.
- BLEU (Bilingual Evaluation Understudy Score): Uses n-gram precision with a brevity penalty.
- ROUGE-N: Calculates n-gram recall between candidate and reference texts.

**Reference-less Evaluation**
- Sentence Similarity: Cosine similarity between sentence embeddings of translated and original English texts, using a multilingual model.


### Directory Structure

```markdown
project-root/
├── data/
│   ├── NSWEduChat_all languages_60 translations[100].xlsx
│   ├── Languages_by_sentence and paragraph.xlsx
│   ├── educhat-translation/           
│   └── post-edited/                   
│       ├── sentence/
│       └── paragraph/
├── results/                           
├── figs/
├── notebooks/
├── src/                               
│   ├── clean_postedits.py
│   ├── compute_eval_scores.py
│   ├── compute_sim_score.py
│   └── process_translations.py
└──README.md                          
```

### Setup
**Python Version:**  `3.11.9`

**Virtual Environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Dependencies:**

```bash
pip install -r requirements.txt
```

### Processing Translations


```bash
python3 src/process_translations.py --input_file "<path_to_input_excel>" --output_dir "<output_directory>" --level <sentence|paragraph>
```

### Clean Post-edited Translations

```bash
python3 src/clean_postedits.py --input_file "<path_to_input_excel>" --output_dir "<output_directory>" --level <sentence|paragraph>
```

### Computing Similarity Scores

```bash
python3 src/compute_eval_scores.py --input_file "<path_to_input_csv>" --output_file "<path_to_json_file>" --use_gpu
```
- Set the flag (`--use_gpu`) to use GPU.

### Computing Evaluation Scores

```bash
python3 src/compute_eval_scores.py --level <sentence|paragraph>
```