import re
import yaml
import torch
import pickle
import datasets
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from pandas import DataFrame
from typing import Optional, List, Tuple
from torch.utils.data import DataLoader
from pylate.indexes import Voyager
from pylate.retrieve import ColBERT

from pmc_clip_utils.pmc_clip import PMC_CLIP
from pmc_clip_utils.dataset import QueryDataset
from bert_whitening import BERTWhitening


class QueryTokenizer:
    def __init__(self, tokenizer, que_len):
        self.tokenizer = tokenizer
        self.que_len = que_len

    def __call__(self, batch):
        # Extract images and questions separately
        image_pt_list = [item['img_pt'] for item in batch]
        question_txt_list = [item['que_txt'] for item in batch]
        # Tokenize questions
        questions = self.tokenizer(
            question_txt_list,
            padding='max_length',
            truncation=True,
            max_length=self.que_len,
            return_tensors='pt'
        )  # (bs, que_len)

        return {
            'img_pt': torch.stack(image_pt_list),
            'que_ids': questions['input_ids'],
            'que_am': questions['attention_mask'],
        }


class DocumentTokenizer:
    def __init__(self, tokenizer, doc_len):
        self.tokenizer = tokenizer
        self.doc_len = doc_len

    def __call__(self, batch):
        # Extract ids and triplets separately
        id_list = [item['id'] for item in batch]
        document_list = [item['triplet'] for item in batch]
        # Tokenize documents
        documents = self.tokenizer(
            document_list,
            padding='max_length',
            truncation=True,
            max_length=self.doc_len,
            return_tensors='pt'
        )  # (bs, doc_len)

        return {
            'id': id_list,
            'doc_ids': documents['input_ids'],
            'doc_am': documents['attention_mask'],
        }


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def load_index_config(input_str: str) -> Optional[Tuple[dict, str]]:
    match = re.match(r'([A-Za-z]+)_E(\d+)M(\d+)', input_str)
    if match:
        return {'embedding_size': int(match.group(2)), 'M': int(match.group(3))}, match.group(1)

    print('ERROR: Could not extract index config!')
    return None

def load_encoder(cfg: dict) -> PMC_CLIP:
    # load model
    model = PMC_CLIP(**cfg['pmc_clip'])
    model.to('cuda')
    print(f'-- RN50_fusion4 Loaded.')
    # load checkpoint
    checkpoint_data = torch.load(cfg['encoder_checkpoint'], weights_only=False)
    state_dict = {k[len('module.'):]: v for k, v in checkpoint_data['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f'-- Checkpoint PMC_CLIP Loaded.')

    return model

def load_pmcvqa_data(csv_path, image_path) -> DataFrame:
    data_df = pd.read_csv(csv_path).dropna(ignore_index=True)
    data_df['Figure_path'] = image_path + data_df['Figure_path']

    return data_df

def load_query_data(
        cfg: dict,
        tokenizer,
        encoder,
        type: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Load queries (questions and images)
    data_df = load_pmcvqa_data(cfg['data_path'] + cfg['test_clean'], cfg['data_path'] + cfg['images'])
    collate_fn = QueryTokenizer(tokenizer, cfg['question_len'])
    test_data = QueryDataset(data_df)
    test_loader = DataLoader(dataset=test_data, collate_fn=collate_fn, **cfg['dataloader'])

    # Encode queries (questions and images)
    return encode_queries(encoder, test_loader, type)

def load_umls_data(docs_path) -> DataFrame:
    return pd.read_csv(docs_path)

# TODO: if no bw exists...
def load_index_bw(index_cfg: dict, name: str) -> Tuple[Voyager, BERTWhitening]:
    # load index
    index = Voyager(index_folder=name, index_name='index', override=False, **index_cfg)
    # load BERT whitening
    with open(rf'{name}/bert_whitening_docs.pkl', 'rb') as f:
        bw = pickle.load(f)

    return index, bw

def load_documents_loader(cfg: dict, tokenizer) -> DataLoader:
    # Load documents
    collate_fn = DocumentTokenizer(tokenizer, cfg['document_len'])
    doc_data = datasets.Dataset.from_pandas(load_umls_data(cfg['documents']))

    return DataLoader(dataset=doc_data, collate_fn=collate_fn, **cfg['dataloader'])

def load_document_data(
        index_cfg: dict,
        encoder,
        data_loader: DataLoader,
        queries_embeddings: List[torch.Tensor],
        name: str,
        type: str,
        bw: BERTWhitening = None
) -> Tuple[Voyager, List[torch.Tensor]]:
    # Initialize index
    index = Voyager(index_folder=name, index_name='index', override=True, **index_cfg)
    # Encode documents
    if bw is not None:
        # Fit BERT whitening with queries
        bw.incremental_fit(queries_embeddings)
        print('-- BERT Whitening Built.')
        # Transform queries with BERT whitening
        queries_embeddings = bw.transform_norm(queries_embeddings)  # [(seq_len, 768)] | [(1, 768)]
        print(f'-- Query whitened, shape: {tuple(queries_embeddings[0].shape)}')
    else:
        # Normalize queries
        queries_embeddings = [torch.nn.functional.normalize(q, dim=-1) for q in queries_embeddings]
        print('-- Query Normalized.')  # [(seq_len, 768)] | [(1, 768)]

    return encode_documents(encoder, data_loader, index, type, bw), queries_embeddings

def get_stage(type: str, data_type: str, bw: bool = True) -> Optional[str]:
    # error handling
    if type not in {'Rep', 'Inter'}:
        print(f'ERROR: Invalid type: {type}!')
    if data_type not in {'query', 'document'}:
        print(f'ERROR: Invalid data_type: {data_type}!')

    messages = {
        ('Rep', 'query', True): 'query_mean',
        ('Rep', 'document', True): 'document_mean',
        ('Rep', 'document', False): 'document_mean_norm',
        ('Inter', 'query', True): 'query',
        ('Inter', 'document', True): 'document',
        ('Inter', 'document', False): 'document_norm',
    }
    return messages[(type, data_type, bw)]

def init_bw(
        encoder,
        data_loader: DataLoader,
        embed_dim: int,
        embedding_size: int,
        type: str
) -> Optional[BERTWhitening]:
    # No BERT whitening
    if embed_dim <= embedding_size:
        return None

    # Fitting loop
    bw = BERTWhitening(embedding_size)
    for batch in tqdm(data_loader):
        embeddings = encoder(batch, get_stage(type, 'document', True))  # [(n, 768)] | [(1, 768)]
        bw.incremental_fit(embeddings)  # Fit BERT whitening with documents
    bw.compute_kernel()
    print('-- BERT whitening model fitted.')

    return bw

def save_bw(bw: BERTWhitening, name: str):
    folder_path = Path(name)
    folder_path.mkdir(exist_ok=True)
    with open(rf'{name}/bert_whitening_docs.pkl', 'wb') as f:
        pickle.dump(bw, f)  # Save BERT whitening

def encode_queries(
        encoder,
        data_loader: DataLoader,
        type: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    images_embeddings, questions_embeddings = [], []
    for batch in data_loader:
        # Encode
        embeddings = encoder(batch, get_stage(type, 'query'))
        # Add the embeddings to the lists
        images_embeddings.extend(embeddings['images_embeddings'])  # [(50, 768)] | [(1, 768)]
        questions_embeddings.extend(embeddings['questions_embeddings'])  # [(que_len, 768)] | [(1, 768)]

    # Return queries embeddings
    return images_embeddings, questions_embeddings

def encode_documents(
        encoder,
        data_loader: DataLoader,
        index: Voyager,
        type: str,
        bw: BERTWhitening = None
) -> Voyager:
    # batch_ids = []
    # batch_embeddings = []
    # count = 0
    #
    # # Encoding loop
    # for batch in tqdm(data_loader):
    #     # Encode
    #     if bw is not None:
    #         embeddings = encoder(batch, get_stage(type, 'document', True))
    #         embeddings = bw.transform_norm(embeddings)  # [(n, embedding_size)] | [(1, embedding_size)]
    #     else:
    #         embeddings = encoder(batch, get_stage(type, 'document', False))  # [(n, 768)] | [(1, 768)]
    #     batch_ids.extend(batch['id'])
    #     batch_embeddings.extend(embeddings)
    #
    #     # Need to be indexed
    #     flag = True
    #
    #     if count % 800 == 0:
    #         # Add the documents ids and embeddings to the Voyager index
    #         index.add_documents(
    #             documents_ids=batch_ids,
    #             documents_embeddings=batch_embeddings
    #         )
    #         print(f'-- Adding the {count} batches to index. length: {len(batch_ids)} {len(batch_embeddings)}')
    #         batch_ids = []
    #         batch_embeddings = []
    #         flag = False
    #
    #     count = count + 1
    #
    # if flag:
    #     # Add the documents ids and embeddings to the Voyager index
    #     index.add_documents(
    #         documents_ids=batch_ids,
    #         documents_embeddings=batch_embeddings
    #     )
    #     print(f'-- Adding the last batches to index. length: {len(batch_ids)} {len(batch_embeddings)}')

    batch_ids = []
    batch_embeddings = []

    # Encoding loop
    for batch in tqdm(data_loader):
        # Encode
        if bw is not None:
            embeddings = encoder(batch, get_stage(type, 'document', True))
            embeddings = bw.transform_norm(embeddings)  # [(n, embedding_size)] | [(1, embedding_size)]
        else:
            embeddings = encoder(batch, get_stage(type, 'document', False))  # [(n, 768)] | [(1, 768)]

        batch_ids.extend(batch['id'])
        batch_embeddings.extend(embeddings)

    print(f'-- Length: {len(batch_ids)} {len(batch_embeddings)}')

    # Add the documents ids and embeddings to the Voyager index
    index.add_documents(
        documents_ids=batch_ids,
        documents_embeddings=batch_embeddings
    )
    print(f'-- All batches added to index.')

    return index

def save_topk_dict(
        scores: list[list[dict]],
        k: int,
        name: str,
        query_type: str
):
    with open(rf'{name}/{query_type}_top{k}.pkl', 'wb') as f:
        pickle.dump(scores, f)

def retrieve_documents(
        cfg: dict,
        embeddings,
        index,
        name: str,
        query_type: str
):
    # Create retriever (late interaction)
    retriever = ColBERT(index=index)
    # Retrieve documents: [[k * {'id': id, 'score': score}]]
    scores = retriever.retrieve(queries_embeddings=embeddings, k=cfg['top_k'])
    # Save results
    save_topk_dict(scores, cfg['top_k'], name, query_type)
