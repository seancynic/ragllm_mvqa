import os
import json
import pandas as pd
import nltk
import ollama
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from typing import List
from tqdm import tqdm


# Tokenization helper function
def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())

# Function to use Ollama LLM for correctness evaluation
def evaluate_with_llm(reference: str, candidate: str) -> str:
    prompt = f"""
    You are an expert evaluator of medical image-based question answering.
    Given a reference answer and a candidate answer, decide if the candidate is correct.
    
    Reference Answer: {reference}
    Candidate Answer: {candidate}

    Reply with ONLY "Correct" or "Wrong".
    """

    response = ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}])
    evaluation = response["message"]["content"].strip()

    return "Correct" if "correct" in evaluation.lower() else "Wrong"

# Function to compute metrics and save results to a CSV file
def compute_metrics(folder_path: str, csv_file: str, output_csv: str):
    """
    Compute BLEU, Precision, Recall, and F1 Score for all JSONL files in a folder against a CSV file,
    and save the results to an output CSV file.

    Parameters:
        folder_path (str): Path to the folder containing JSONL files.
        csv_file (str): Path to the CSV file containing ground truth answers.
        output_csv (str): Path to save the output CSV file with results.
    """
    # Load CSV file
    df = pd.read_csv(csv_file)

    # Get all JSONL files in the folder
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

    # List to store results
    results_list = []

    for jsonl_file in jsonl_files:
        jsonl_path = os.path.join(folder_path, jsonl_file)

        # Load JSONL file
        json_list = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                json_list.append(json.loads(line.strip()))

        # Ensure JSONL and DataFrame have the same length
        assert len(json_list) == len(df), f"Mismatch between {jsonl_file} and DataFrame length."

        # Initialize metric lists
        bleu_scores = []
        precisions = []
        recalls = []
        f1_scores = []
        llm_evaluations = []

        # List to store row-wise results
        row_results = []

        # Compute metrics for each pair
        for i, entry in tqdm(enumerate(json_list)):

            # Get reference (ground truth) and candidate (model output)
            reference = df.iloc[i]["Answer"].strip()
            candidate = entry["text"].strip()

            # Tokenize
            ref_tokens = tokenize(reference)
            cand_tokens = tokenize(candidate)

            # Compute BLEU Score (with smoothing for short sentences)
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)

            # Compute Precision, Recall, and F1 Score
            ref_counter = Counter(ref_tokens)
            cand_counter = Counter(cand_tokens)

            true_positive = sum((ref_counter & cand_counter).values())
            false_positive = sum((cand_counter - ref_counter).values())
            false_negative = sum((ref_counter - cand_counter).values())

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0  # To avoid division by zero

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            # Use LLM to evaluate correctness
            llm_result = evaluate_with_llm(reference, candidate)
            llm_evaluations.append(llm_result)

            # Append row-wise results
            row_results.append({
                "Question": df.iloc[i]["Question"].strip(),
                "Reference Answer": reference,
                "Candidate Answer": candidate,
                "LLM Evaluation": llm_result,
            })

        # Convert results to a DataFrame and save as CSV
        results_df = pd.DataFrame(row_results)
        results_df.to_csv(f"{folder_path}/{jsonl_file[7:-6]}_LLMEval.csv", index=False)

        print(f"Results saved to {jsonl_file[7:-6]}_LLMEval.csv")

        # Compute macro mean
        macro_bleu = sum(bleu_scores) / len(bleu_scores)
        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        macro_f1 = sum(f1_scores) / len(f1_scores)
        llm_accuracy = llm_evaluations.count("Correct") / len(llm_evaluations)

        # Append results to list
        results_list.append({
            "JSONL File": jsonl_file[:-6],
            "Macro BLEU Score": macro_bleu,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1 Score": macro_f1,
            "LLM Correct Count": llm_evaluations.count("Correct"),
            "LLM Accuracy": llm_accuracy,
            "Total Number": len(llm_evaluations),
        })

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")


folder = "Inter_E128M64"

folder_path = rf"../llava_json/pmc_oa_captions/{folder}/answer"  # Replace with the actual folder path
csv_file = rf"../test_clean_radiology.csv"  # Replace with the actual CSV file path
output_csv = rf"../llava_json/pmc_oa_captions/{folder}/metrics_results.csv"  # Output file name

# filter_path = rf"../choose_radio/idx_not_radio.pkl"
# with open(filter_path, "rb") as f:
#     filter = pickle.load(f)

compute_metrics(folder_path, csv_file, output_csv)