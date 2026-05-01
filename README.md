# ragllm_mvqa

## rag_faiss

### preprocess raw umls data:

1.  **Convert to CSV**

    Convert raw umls data to csv file using UMLS.py.

2.  **Preprocess (Data Clean and Selection)**

    Clean and select certain data using preprocess.ipynb.

### ragllm steps:

1.  **Generate Embeddings**

    Convert query data into vector representations using generate_query_embeds.py.

3.  **Retrieve Context**

    Perform a similarity search using retrieve_cb.py.

5.  **Prepare Model Input**

    Format the retrieved information and image metadata into the standard input JSON format required by LLaVA-Med using make_json_data.py.

7.  **Inference**

    Run the LLaVA-Med model using the json files generated in the previous step.

9.  **Evaluation**

    After obtaining the model outputs, calculate performance metrics using eval.py.
