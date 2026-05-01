# ragllm_mvqa

## rag_faiss

### Workflow Steps

1.  **Generate Embeddings**
    First, use generate_query_embeds.py convert your query data into vector representations.

2.  **Retrieve Context**
    Perform a similarity search using retrieve_cb.py.

3.  **Prepare Model Input**
    Format the retrieved information and image metadata into the standard input JSON format required by LLaVA-Med using make_json_data.py.

4.  **Inference**
    Run the LLaVA-Med model using the json files generated in the previous step.

5.  **Evaluation**
    After obtaining the model outputs, calculate performance metrics using eval.py.
    ```
