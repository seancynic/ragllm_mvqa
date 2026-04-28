import faiss
import pickle
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.decomposition import IncrementalPCA

from pmc_clip_utils.pmc_clip import PMC_CLIP
from pmc_clip_utils.config import model_cfg


pmcclip_checkpoint = 'pmc_clip_utils/checkpoint.pt'
data_path = 'umls_datasets/'
triplets_path = f'{data_path}UMLS_RELA_posDEF.csv'
# triplets_path = 'UMLS_REL+RELA/UMLS_REL_posDEF.csv'
index_path = f'{data_path}index_new3/'
query_path = f'{data_path}query_embeds/'
# query_path = ''
log_path = '/users/xwu20/Medical/logs/'

entity_length = 30
relation_length = 16
triplet_length = entity_length + relation_length + entity_length
PCA_dim = 32
embedding_dim = 768
batch_size = 256


def load_encoder(cp):
    # load model
    model = PMC_CLIP(**model_cfg)
    model.to('cuda')
    print(f'-- RN50_fusion4 Loaded.')
    # load checkpoint
    checkpoint_data = torch.load(cp)
    state_dict = checkpoint_data['state_dict']
    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f'-- Checkpoint PMC_CLIP Loaded.')
    return model

def normalize(arr, dim):
    return arr / np.linalg.norm(arr, axis=dim, keepdims=True)

def get_concat_embeds(encoder, tokenizer, head, rel, tail, gpu):
    inputs_head = tokenizer(head,
                            padding='max_length',
                            truncation=True,
                            max_length=entity_length,
                            return_tensors='pt')  # (bs, 30)
    inputs_rel = tokenizer(rel,
                           padding='max_length',
                           truncation=True,
                           max_length=relation_length,
                           return_tensors='pt')  # (bs, 16)
    inputs_tail = tokenizer(tail,
                            padding='max_length',
                            truncation=True,
                            max_length=entity_length,
                            return_tensors='pt')  # (bs, 30)
    input_ids = torch.cat((inputs_head['input_ids'], inputs_rel['input_ids'], inputs_tail['input_ids']), dim=-1)
    attention_mask = torch.cat((inputs_head['attention_mask'], inputs_rel['attention_mask'], inputs_tail['attention_mask']), dim=-1)
    embeds = encoder.encode_text_all(input_ids.to('cuda'), attention_mask.to('cuda'))
    if gpu:
        return embeds  # (bs, triplet_length, embedding_dim)
    else:
        return embeds.cpu().detach().numpy().astype(np.float32)  # (bs, triplet_length, embedding_dim)

def get_embeds(encoder, tokenizer, data_df, slice, gpu):
    if slice == False:
        batch_head = data_df['CUI1'].tolist()
        batch_rel = data_df['REL'].tolist()
        batch_tail = data_df['CUI2'].tolist()
    else:
        batch_head = data_df['CUI1'][slice[0]:slice[1]].tolist()
        batch_rel = data_df['REL'][slice[0]:slice[1]].tolist()
        batch_tail = data_df['CUI2'][slice[0]:slice[1]].tolist()
    return get_concat_embeds(encoder, tokenizer, batch_head, batch_rel, batch_tail, gpu)

def build_training_index(xb, type):
    index = faiss.index_factory(PCA_dim, type, faiss.METRIC_INNER_PRODUCT)
    index.train(xb)
    index.add(xb)
    return index

def build_nontraining_index(xb, type):
    index = faiss.index_factory(PCA_dim, type, faiss.METRIC_INNER_PRODUCT)
    index.add(xb)
    return index

def load_query(filename):
    xq = np.load(query_path + filename)  # (num_sample, img_length, embedding_dim)
    return xq.reshape(-1, embedding_dim), xq  # (num_sample * img_length, embedding_dim)

def stage1_faiss(index, token_xq, k, seq_length):
    D, I = index.search(token_xq, k)  # (num_sample * seq_length, k)
    return I.reshape((-1, seq_length * k)) // triplet_length  # (num_sample, seq_length * k)

def maxsim_sum(q, d):
    '''
    Args:
        q: (seq_length, embedding_dim)
        d: (nd, triplet_length, embedding_dim)
    Returns:
        sims: (1, nd)
    '''
    tmp = torch.max(torch.tensordot(q, d, dims=[[-1], [-1]]), dim=-1)
    return torch.sum(tmp[0], dim=0, keepdim=True)

def stage2_colbert(encoder, tokenizer, data_df, xq, k, indices):
    # init
    stage2_D = np.zeros((0, k))
    stage2_I = np.zeros((0, k))
    xq_norm = F.normalize(xq, dim=-1)
    # traverse all queries
    for q, idx in zip(xq_norm, indices):
        # get selected docs
        selected_docs = data_df.iloc[idx]
        # get embeds of selected docs
        d = torch.zeros((0, triplet_length, embedding_dim), device='cuda')
        for i in range(0, len(selected_docs), batch_size):
            docs = get_embeds(encoder, tokenizer, selected_docs, (i, i + batch_size), True)
            docs = F.normalize(docs, dim=-1)
            d = torch.cat([d, docs], dim=0)  # (seq_length * k, triplet_length, 768)

        # MaxSimSum
        dis = maxsim_sum(q, d)  # (1, seq_length * k)
        # top k sort in descending order
        D, I = torch.sort(dis, dim=-1, descending=True)
        topk_D = D[:, :k].cpu().detach().numpy()  # (1, k)
        topk_I = idx[I[:, :k].cpu().detach().numpy()]  # (1, k)

        # concat
        stage2_D = np.concatenate([stage2_D, topk_D], axis=0)  # (num_sample, k)
        stage2_I = np.concatenate([stage2_I, topk_I], axis=0)  # (num_sample, k)

    return stage2_D, stage2_I  # (num_sample, k) in descending order

def retrieve(encoder, tokenizer, data_df,
             xq, token_xq, token_xb, seq_length,
             index_type, index_name):
    # stage 1: FAISS retrieval (CPU)
    # build index
    if 'IVF' in index_type:
        index = build_training_index(token_xb, index_type)
    elif 'PQ' in index_type:
        index = build_training_index(token_xb, index_type)
    elif 'PCA' in index_type:
        index = build_training_index(token_xb, index_type)
    else:
        index = build_nontraining_index(token_xb, index_type)
    faiss.write_index(index, f'{index_path}{index_name}.index')
    # retrieve
    k1 = 20
    stage1_I = stage1_faiss(index, token_xq, k1, seq_length)  # (num_sample, seq_length * k)
    stage1_uniqueI = [np.unique(sample[sample != -1]) for sample in stage1_I]
    print('-- Stage 1 Finished.')
    with open(f'{log_path}Stage1-{index_name}', 'wb') as fp:
        pickle.dump([1, 1], fp)

    # stage 2: ColBERT retrieval (GPU)
    k2 = 10
    stage2_D, stage2_I = stage2_colbert(encoder, tokenizer, data_df, torch.tensor(xq, dtype=torch.float32, device='cuda'), k2, stage1_uniqueI)
    print('-- Stage 2 Finished.')
    with open(f'{log_path}Stage2-{index_name}', 'wb') as fp:
        pickle.dump([1, 1], fp)

    return stage2_D, stage2_I


if __name__ == '__main__':
    # load model
    encoder = load_encoder(pmcclip_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['text_cfg']['bert_model_name'])
    ipca = IncrementalPCA(n_components=PCA_dim)

    # load data
    data_df = pd.read_csv(triplets_path, keep_default_na=False)
    img_tokens_xq, img_xq = load_query('image_embeds_PC.npy')
    print('-- data_df, xq, tokens_xq Loaded.')

    # q -> fit PCA
    ipca.partial_fit(img_tokens_xq)
    print('-- q -> fit PCA.')
    with open(f'{log_path}q_fit_PCA', 'wb') as fp:
        pickle.dump([1, 1], fp)

    # xb -> fit PCA
    count = 0
    for i in range(0, len(data_df), batch_size):
        triplets = get_embeds(encoder, tokenizer, data_df, (i, i + batch_size), False)  # (bs, triplet_length, embedding_dim)
        ipca.partial_fit(triplets.reshape(-1, embedding_dim))
        # count bar
        count += 1
        if count % 100 == 0:
            with open(f'{log_path}xb_fit_PCA_{count}', 'wb') as fp:
                pickle.dump([1, 1], fp)
    joblib.dump(ipca, f'{index_path}IPCA{PCA_dim}.pkl')
    print('-- xb -> fit PCA.')
    with open(f'{log_path}xb_fit_PCA', 'wb') as fp:
        pickle.dump([1, 1], fp)

    # init
    tokens_xb = np.zeros((0, PCA_dim))
    count = 0
    # get tokens_xb
    for i in range(0, len(data_df), batch_size):
        triplets = get_embeds(encoder, tokenizer, data_df, (i, i + batch_size), False)
        # generate dim-reduced embeds
        batch_tokens_xb = ipca.transform(triplets.reshape(-1, embedding_dim))  # (bs * triplet_length, PCA_dim)
        # generate xb
        tokens_xb = np.concatenate([tokens_xb, batch_tokens_xb], axis=0)  # (2.6M * triplet_length, PCA_dim)
        # count bar
        count += 1
        if count % 100 == 0:
            with open(f'{log_path}get_tokens_xb_{count}', 'wb') as fp:
                pickle.dump([1, 1], fp)
    print(f'tokens_xb: {tokens_xb.shape}')
    print('-- Get tokens_xb.')
    with open(f'{log_path}get_tokens_xb', 'wb') as fp:
        pickle.dump([1, 1], fp)
    # get tokens_xq
    tokens_xq = ipca.transform(img_tokens_xq)  # (num_sample, img_length, PCA_dim)
    print('-- Get tokens_xq.')
    with open(f'{log_path}get_tokens_xq', 'wb') as fp:
        pickle.dump([1, 1], fp)

    # retrieval
    D, I = retrieve(encoder, tokenizer, data_df,
                    img_xq, tokens_xq, tokens_xb, img_xq.shape[1],
                    index_type='IVF1600,Flat', index_name=f'PC_IPCA{PCA_dim}_IVF')  # (num_sample, k)
    print('-- IVF Retrieved.')
    np.save(f'{index_path}PC_IPCA{PCA_dim}_IVF_D10.npy', D)
    np.save(f'{index_path}PC_IPCA{PCA_dim}_IVF_I10.npy', I)

    D, I = retrieve(encoder, tokenizer, data_df,
                    img_xq, tokens_xq, tokens_xb, img_xq.shape[1],
                    index_type='HNSW', index_name=f'PC_IPCA{PCA_dim}_HNSW')  # (num_sample, k)
    print('-- HNSW Retrieved.')
    np.save(f'{index_path}PC_IPCA{PCA_dim}_HNSW_D10.npy', D)
    np.save(f'{index_path}PC_IPCA{PCA_dim}_HNSW_I10.npy', I)
