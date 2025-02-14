#!/usr/bin/env python3
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate similarity scores between translations using LaBSE."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input CSV file containing translations.",
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the output JSON file to store results."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Flag to use GPU/MPS if available. If not set, CPU is used.",
    )
    return parser.parse_args()


def setup_device(use_gpu_flag):
    if use_gpu_flag:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    # Default to CPU if use_gpu flag not set or no GPU available.
    return torch.device("cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embeds(sentences, tokenizer, model, device):
    # Ensure all inputs are strings (handling NaN)
    sentences = [str(s) if pd.notnull(s) else "" for s in sentences]
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def get_similarity(embed_1, embed_2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(embed_1, embed_2)[0].item()


def get_sim_score(data, tokenizer, model, device):
    cols = data.columns
    sim_scores = {lang: [] for lang in cols[1:]}
    for i in tqdm(range(len(data)), desc="Processing rows"):
        try:
            # Convert row to a list of sentences
            row_sentences = list(data.iloc[i][cols[:]])
            embeds = get_embeds(row_sentences, tokenizer, model, device)
            for j, key in enumerate(sim_scores.keys()):
                # Compare first column with each other column (j+1)
                sim_scores[key].append(get_similarity(embeds[[0]], embeds[[j + 1]]))
        except Exception as e:
            print(f"Error on row {i}: {e}")
            continue
    return sim_scores


def main():
    args = parse_arguments()
    device = setup_device(args.use_gpu)
    print(f"Using device: {device}")

    # Load tokenizer and model and move model to the device.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    model.to(device)

    # Read the CSV file
    data = pd.read_csv(args.input_file)

    # Compute similarity scores
    sim_scores = get_sim_score(data, tokenizer, model, device)
    # Create a DataFrame, calculate the mean score for each language,
    # and format the result.
    sim_score_df = pd.DataFrame(sim_scores).mean().reset_index()
    sim_score_df.columns = ["language", "score"]

    # Write out the result to a JSON file.
    sim_score_df.to_json(args.output_file, orient="records", indent=4)
    print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
