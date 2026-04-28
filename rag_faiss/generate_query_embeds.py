import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pmc_clip_utils.pmc_clip import PMC_CLIP
from pmc_clip_utils.dataset import PMCVQADatasetRaw
from pmc_clip_utils.config import model_cfg


query_path = '../PMC_VQA/'
query_embeds_path = 'query_embeds/'
pmcclip_checkpoint = 'pmc_clip_utils/checkpoint.pt'
dataset = PMCVQADatasetRaw
embedding_dim = 768
batch_size = 128
n_workers = 0
single_file_size = 2 ** 11

# query_path = '../PMC-VQA/'
# doc_path = 'weights/'
# pmcclip_checkpoint = 'pmc_clip_utils/checkpoint.pt'
# dataset = PMCVQADatasetRaw
# embedding_dim = 768
# batch_size = 32
# n_workers = 0


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

def load_csv(csv_file):
    data_df = pd.read_csv(query_path + csv_file).dropna(ignore_index=True)
    data_df['Figure_path'] = query_path + 'images/' + data_df['Figure_path']
    return data_df

def get_all_query(encoder, tokenizer, data_df):
    # separate embed files
    count = 0
    for i in range(0, len(data_df), single_file_size):
        # load data for a single file
        test = dataset(data_df[i:i + single_file_size], tokenizer, model_cfg['text_cfg']['context_length'], 'test')
        test_data = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        # init
        img_qs, que_qs = np.zeros((0, 50, 768)), np.zeros((0, 77, 768))

        # generate and concat
        for batch in test_data:
            mats = encoder(batch, 'img_que_mat')  # (bs, 50, 768), (bs, 77, 768)
            img_qs = np.concatenate([img_qs, mats['image_mats'].cpu().detach().numpy()], axis=0)
            que_qs = np.concatenate([que_qs, mats['question_mats'].cpu().detach().numpy()], axis=0)

        # save query
        np.save(query_embeds_path + f'image_embeds_{count:02}.npy', img_qs)
        np.save(query_embeds_path + f'question_embeds_{count:02}.npy', que_qs)

        count += 1


if __name__ == '__main__':
    # load encoder
    encoder = load_encoder(pmcclip_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['text_cfg']['bert_model_name'])

    # load dataset
    data_df = load_csv('test_clean.csv')

    # get query
    get_all_query(encoder, tokenizer, data_df)
    print('-- get query')
