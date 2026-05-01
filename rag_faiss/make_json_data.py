import numpy as np
import pandas as pd
import json

type = 'PC_avg_IVFPQ'

# topk_idx_que = np.load(f'topk_idx_avg/topk_idx_que_{type}.npy')
topk_idx_img = np.load(f'topk_idx_avg/topk_idx_img_{type}.npy')
# topk_idx_img = np.load(f'PC_IPCA32_IVF_I10.npy').astype(np.int_)

umls = pd.read_csv('UMLS_REL+RELA/UMLS_REL_posDEF.csv', keep_default_na=False)
test = pd.read_csv('../SOURCE/PMC-VQA/test_clean.csv')[['Figure_path', 'Question', 'Answer']]

def oper(row):
    first_col = row.iloc[0].strip()[:-1] if row.iloc[0].strip().endswith('.') else row.iloc[0].strip()
    second_col = row.iloc[1].strip()[:-1] if row.iloc[1].strip().endswith('.') else row.iloc[1].strip()
    third_col = row.iloc[2].strip()[:-1] if row.iloc[2].strip().endswith('.') else row.iloc[2].strip()
    return f'head entity: {first_col}; relation: {second_col}; tail entity: {third_col}.'

# def oper(row):
#     return f'{row.iloc[0]} {row.iloc[1]} {row.iloc[2]}.'

concat_tri = umls.apply(lambda row: oper(row), axis=1)

# concat_tri = umls.apply(lambda row: ' '.join(row), axis=1)

for i in range(0, 10):
    test[f'top{i + 1}_img'] = concat_tri[topk_idx_img[:, i]].reset_index(drop=True)
    # test[f'top{i + 1}_que'] = concat_tri[topk_idx_que[:, i]].reset_index(drop=True)

# Initialize an empty list to store JSON objects
json_data = []
# Iterate over each row in the DataFrame
for index, row in test.iterrows():
    # make prompt
    context_str = f"1.{row['top1_img']} 2.{row['top2_img']} 3.{row['top3_img']} 4.{row['top4_img']} 5.{row['top5_img']}"
    # context_str = f"{row['top1_img']} {row['top2_img']} {row['top3_img']}"
    query_str = row['Question'].strip()
    prompt = (
        f"The retrieved background knowledge is given as follows: {context_str} "
        f"Please answer the question: {query_str}"
    )
    # prompt = (
    # "Context information is below.\n"
    # "---------------------\n"
    # f"{context_str}\n"
    # "---------------------\n"
    # f"Using both the context information and also your own knowledge, answer the question: {query_str}\n"
    # "If the context isn't helpful, you can also answer the question on your own.\n"
    # )
    # Create the JSON object
    json_obj = {
        "id": row['Figure_path'].split('.')[0],  # Extracting the image name from the path
        "image": row['Figure_path'],
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": row['Answer'].strip()
            }
        ]
    }
    # Append the JSON object to the list
    json_data.append(json_obj)
# Write the list of JSON objects to a JSON file
with open(f'llava_json_data/{type}/test_clean_top5_img.json', 'w') as f:
    json.dump(json_data, f, indent=2)
# with open(f'llava_json_data/test_clean_32top10_img_v1.json', 'w') as f:
#     json.dump(json_data, f, indent=2)