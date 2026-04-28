import torch
from transformers import AutoTokenizer

from retrieval_utils import (
    load_config,
    load_index_config,
    load_encoder,
    load_query_data,
    load_document_data,
    load_documents_loader,
    init_bw,
    save_bw,
    retrieve_documents
)


def main():
    import os
    import wandb
    experiment_name = os.environ.get('EXPERIMENT_NAME', None)
    wandb.login(key=os.environ.get('WANDB_API_KEY', None))
    wandb.init(
        project=os.environ.get('PROJECT_NAME', None),
        name=experiment_name,
    )
    # experiment_name = 'Rep_E768M64'

    # Load config, index config, tokenizer and encoder
    cfg = load_config('retrieval_config.yaml')
    index_cfg, retrieval_type = load_index_config(experiment_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg['text_encoder'])
    encoder = load_encoder(cfg)
    # index = load_index_bw(index_cfg, experiment_name)

    # Load query embeddings
    images_embeddings, questions_embeddings = load_query_data(cfg, tokenizer, encoder, retrieval_type)
    query_embeddings = [torch.cat((ie, qe), dim=0) for ie, qe in zip(images_embeddings, questions_embeddings)]
    print(f'  Number of images: {len(images_embeddings)}, shape: {tuple(images_embeddings[0].shape)}')
    print(f'  Number of questions: {len(questions_embeddings)}, shape: {tuple(questions_embeddings[0].shape)}')
    print(f'  Number of queries: {len(query_embeddings)}, shape: {tuple(query_embeddings[0].shape)}')
    print('*** Query Encoded ***')

    # Load documents loader
    umls_loader = load_documents_loader(cfg, tokenizer)

    # Initialize BERT whitening with documents
    bw = init_bw(
        encoder, umls_loader, cfg['pmc_clip']['embed_dim'], index_cfg['embedding_size'], retrieval_type)

    # Load document embeddings
    if bw is not None:
        # Save and copy BERT whitening
        save_bw(bw, experiment_name)
        # Load document embeddings
        index, query_embeddings = load_document_data(
            index_cfg, encoder, umls_loader, query_embeddings, experiment_name, retrieval_type, bw)
        print('-- Index Built.')
        # Retrieval
        retrieve_documents(cfg, query_embeddings, index, experiment_name, 'image_question')
        print('*** Retrieved with Images + Questions ***')
    else:
        # Load document embeddings
        index, query_embeddings = load_document_data(
            index_cfg, encoder, umls_loader, query_embeddings, experiment_name, retrieval_type)
        print('-- Index Built.')
        # Retrieval
        retrieve_documents(cfg, query_embeddings, index, experiment_name, 'image_question')
        print('*** Retrieved with Images + Questions ***')


if __name__ == '__main__':
    main()
