import json
import pickle
import pandas as pd
from pathlib import Path


question_df = pd.read_csv('../test_clean.csv')
document_df = pd.read_csv('../umls_datasets/pmc_oa_captions.csv')

def retrieve_form_json(similarity_list, topk, question_type, retrieve_type, retrieve_results, retrieve_mark):
    Path(f"../llava_json/pmc_oa_captions/{retrieve_results}").mkdir(exist_ok=True)
    Path(f"../llava_json/pmc_oa_captions/{retrieve_results}/question").mkdir(exist_ok=True)
    Path(f"../llava_json/pmc_oa_captions/{retrieve_results}/answer").mkdir(exist_ok=True)

    # Open the JSONL file in write mode
    with open(f"../llava_json/pmc_oa_captions/{retrieve_results}/question/question_{question_type}.jsonl", "w") as jsonl_file:

        # Loop through each row in the question dataframe
        for i, question_row in question_df.iterrows():

            if retrieve_mark:
                # No retrieve for "Retrieve_mark" False
                if not question_row['Retrieve_mark']:
                    # Create the final JSON object for this row
                    final_obj = {
                        "question_id": i,
                        "image": question_row["Figure_path"],
                        "text": f'{question_row["Question"].strip()}\n<image>',
                        # Join all triplets and append the question
                    }
                    # Write the final_obj as a JSON string to the JSONL file
                    jsonl_file.write(json.dumps(final_obj) + "\n")

                    continue

            if retrieve_type:
                # Retrieve the corresponding triplets for the sorted similarity list
                triplet_texts = []
                for item in similarity_list[i]:
                    idx = int(item["id"][1:])
                    # Find the corresponding triplet by the 'idx' in the document dataframe
                    triplet_row = document_df.iloc[idx, :]
                    if not triplet_row.empty:
                        triplet_text = triplet_row['triplet']
                        triplet_texts.append(triplet_text)

                # Create the final JSON object for this row
                final_retrieved_obj = {
                    "question_id": i,
                    "image": question_row["Figure_path"],
                    "text": f'The retrieved background knowledge is given as follows: {". ".join(triplet_texts[:topk])}. '
                            f'{question_row["Question"].strip()}\n<image>',  # Join all triplets and append the question
                }
                # Write the final_obj as a JSON string to the JSONL file
                jsonl_file.write(json.dumps(final_retrieved_obj) + "\n")

            else:
                # Create the final JSON object for this row
                final_obj = {
                    "question_id": i,
                    "image": question_row["Figure_path"],
                    "text": f'{question_row["Question"].strip()}\n<image>',
                    # Join all triplets and append the question
                }
                # Write the final_obj as a JSON string to the JSONL file
                jsonl_file.write(json.dumps(final_obj) + "\n")


retrieve_mark = False
retrieve_type = True
retrieve_results = 'Inter_E128M64'

with open(f'../retrieve_results/pmc_oa_captions/{retrieve_results}/image_question_top10.pkl', 'rb') as file:
    query_top10 = pickle.load(file)

for k in [1, 2, 3, 4, 5]:
    retrieve_form_json(query_top10, k, f'{retrieve_results}IQ{k}', retrieve_type, retrieve_results, retrieve_mark)