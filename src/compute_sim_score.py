#!/usr/bin/env python3
"""
LaBSE Similarity Score Calculator
---------------------------------
This script calculates similarity scores between translations using the LaBSE model.
It reads a CSV file containing translations (with the first column as the reference
and subsequent columns as translations), computes the cosine similarity between the
reference and each translation, and then writes the averaged similarity scores per
language to a JSON file.

Usage:
    python script_name.py --input_file <path_to_csv> --output_file <path_to_output_json> [--use_gpu]

Arguments:
    --input_file  : Path to the input CSV file containing translations.
    --output_file : Path to the output JSON file to store the similarity results.
    --use_gpu     : Optional flag to use GPU/MPS if available; otherwise, CPU is used.

Dependencies:
    - argparse
    - pandas
    - tqdm
    - torch
    - transformers
    - sentence-transformers/LaBSE (for the model and tokenizer)
"""

import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments including:
            --input_file (str): Path to the CSV file containing translations.
            --output_file (str): Path to the JSON file for output results.
            --use_gpu (bool): Flag indicating whether to use GPU/MPS if available.
    """
    parser = argparse.ArgumentParser(
        description="Calculate similarity scores between translations using LaBSE."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input CSV file containing translations.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output JSON file to store results.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Flag to use GPU/MPS if available. If not set, CPU is used.",
    )
    return parser.parse_args()


def setup_device(use_gpu_flag):
    """
    Setup the device for torch computations based on availability and flag.

    Args:
        use_gpu_flag (bool): Whether to attempt to use a GPU/MPS device.

    Returns:
        torch.device: The device to use (cuda, mps, or cpu).
    """
    if use_gpu_flag:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    # Default to CPU if GPU/MPS is not available or flag is not set.
    return torch.device("cpu")


def mean_pooling(model_output, attention_mask):
    """
    Apply mean pooling to the token embeddings from the model output.

    This function computes the average of token embeddings, weighted by the attention mask.

    Args:
        model_output (tuple): Output from the model, where the first element contains token embeddings.
        attention_mask (torch.Tensor): Attention mask indicating valid tokens.

    Returns:
        torch.Tensor: Sentence embeddings computed by mean pooling.
    """
    # Extract token embeddings (first element of model output)
    token_embeddings = model_output[0]
    # Expand the attention mask to match the dimensions of token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    # Compute the weighted sum and divide by the number of valid tokens
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embeds(sentences, tokenizer, model, device):
    """
    Compute normalized sentence embeddings for a list of sentences.

    Args:
        sentences (list): List of sentences (strings) to embed.
        tokenizer (AutoTokenizer): Tokenizer for the LaBSE model.
        model (AutoModel): LaBSE model to generate embeddings.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Normalized sentence embeddings.
    """
    # Convert all sentences to strings and handle NaN values by converting them to empty strings.
    sentences = [str(s) if pd.notnull(s) else "" for s in sentences]
    # Tokenize input sentences with padding and truncation
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    # Move inputs to the specified device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    # Obtain model outputs without gradient computations
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Apply mean pooling and normalize the embeddings
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def get_similarity(embed_1, embed_2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embed_1 (torch.Tensor): Embedding tensor for the first sentence.
        embed_2 (torch.Tensor): Embedding tensor for the second sentence.

    Returns:
        float: Cosine similarity score.
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # Return the similarity score as a Python float
    return cos(embed_1, embed_2)[0].item()


def get_sim_score(data, tokenizer, model, device):
    """
    Compute similarity scores for each row in the data.

    For each row, the function calculates the cosine similarity between the first
    column (reference) and each subsequent translation column.

    Args:
        data (pd.DataFrame): DataFrame containing translations. The first column is the reference.
        tokenizer (AutoTokenizer): Tokenizer for the LaBSE model.
        model (AutoModel): LaBSE model to generate embeddings.
        device (torch.device): Device for computation.

    Returns:
        dict: A dictionary where keys are language names (columns from second onward) and
              values are lists of similarity scores.
    """
    cols = data.columns
    # Initialize similarity score storage for each language column (excluding the first)
    sim_scores = {lang: [] for lang in cols[1:]}
    for i in tqdm(range(len(data)), desc="Processing rows"):
        try:
            # Get the list of sentences for the current row (reference and translations)
            row_sentences = list(data.iloc[i][cols[:]])
            # Compute embeddings for all sentences in the row
            embeds = get_embeds(row_sentences, tokenizer, model, device)
            # Calculate similarity: reference (first column) with each translation (other columns)
            for j, key in enumerate(sim_scores.keys()):
                sim_scores[key].append(get_similarity(embeds[[0]], embeds[[j + 1]]))
        except Exception as e:
            print(f"Error on row {i}: {e}")
            continue
    return sim_scores


def main():
    """
    Main function to compute similarity scores between translations.

    This function:
      1. Parses command-line arguments.
      2. Sets up the computation device.
      3. Loads the LaBSE tokenizer and model.
      4. Reads the input CSV file containing translations.
      5. Computes similarity scores for each translation against the reference.
      6. Aggregates the results and writes the mean similarity score per language to a JSON file.
    """
    # Parse command-line arguments
    args = parse_arguments()
    # Setup the computation device (GPU/MPS or CPU)
    device = setup_device(args.use_gpu)
    print(f"Using device: {device}")

    # Load the tokenizer and model for LaBSE, then move the model to the selected device.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    model.to(device)

    # Read the input CSV file containing translation data.
    data = pd.read_csv(args.input_file)

    # Compute similarity scores between the reference translation and other translations.
    sim_scores = get_sim_score(data, tokenizer, model, device)
    # Aggregate the similarity scores by computing the mean score for each language.
    sim_score_df = pd.DataFrame(sim_scores).mean().reset_index()
    sim_score_df.columns = ["language", "score"]

    # Write the aggregated similarity scores to the output JSON file.
    sim_score_df.to_json(args.output_file, orient="records", indent=4)
    print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
