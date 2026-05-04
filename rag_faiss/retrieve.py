import faiss
import numpy as np


query_embeds_path = '/scratch/xwu20/Medical/query_embeds/'
index_path = '/scratch/xwu20/Medical/index_avg/'
topk_idx_path = '/scratch/xwu20/Medical/topk_idx_avg/'
embedding_dim = 768
k = 10


def get_topk_triplets_idx(index, xq, seq_length):
    D, I = index.search(xq, k)  # (bs * seq_length, k)
    D_reshape = D.reshape((-1, seq_length * k))  # (bs,  seq_length * k)
    I_reshape = I.reshape((-1, seq_length * k))  # (bs,  seq_length * k)
    topk_dis_idx = np.argsort(-D_reshape, axis=-1)[:, :k]  # (bs, k) in descending order
    topk_triplets_idx = I_reshape[np.arange(I_reshape.shape[0])[:, None], topk_dis_idx]  # (bs, k)
    return topk_triplets_idx


if __name__ == '__main__':
    # load query
    img_xq = np.load(query_embeds_path + 'image_embeds_PC.npy')  # (2000, 50, 768)
    que_xq = np.load(query_embeds_path + 'question_embeds_PC.npy')  # (2000, 77, 768)
    print(img_xq.shape)
    print(que_xq.shape)
    print('-- get query')
    img_tokens_xq = np.reshape(img_xq, (-1, embedding_dim))  # (bs * seq_length, embedding_dim)
    que_tokens_xq = np.reshape(que_xq, (-1, embedding_dim))  # (bs * seq_length, embedding_dim)
    print(img_tokens_xq.shape)
    print(que_tokens_xq.shape)
    print('-- get tokens query')
    # # load doc index
    # PC_avg_IVFPQ_index = faiss.read_index(index_path + 'PC_avg_IVFPQ.index')
    # PC_avg_IVFPQ_pca_index = faiss.read_index(index_path + 'PC_avg_IVFPQ_pca.index')
    # PCTH_avg_IVFPQ_index = faiss.read_index(index_path + 'PCTH_avg_IVFPQ.index')
    # PCTH_avg_IVFPQ_pca_index = faiss.read_index(index_path + 'PCTH_avg_IVFPQ_pca.index')
    # print('-- get index')
    # # search
    # topk_idx_img = get_topk_triplets_idx(PC_avg_IVFPQ_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    # topk_idx_que = get_topk_triplets_idx(PC_avg_IVFPQ_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    # np.save(topk_idx_path + 'topk_idx_img_PC_avg_IVFPQ.npy', topk_idx_img)
    # np.save(topk_idx_path + 'topk_idx_que_PC_avg_IVFPQ.npy', topk_idx_que)
    #
    # topk_idx_img = get_topk_triplets_idx(PC_avg_IVFPQ_pca_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    # topk_idx_que = get_topk_triplets_idx(PC_avg_IVFPQ_pca_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    # np.save(topk_idx_path + 'topk_idx_img_PC_avg_IVFPQ_pca.npy', topk_idx_img)
    # np.save(topk_idx_path + 'topk_idx_que_PC_avg_IVFPQ_pca.npy', topk_idx_que)
    #
    # topk_idx_img = get_topk_triplets_idx(PCTH_avg_IVFPQ_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    # topk_idx_que = get_topk_triplets_idx(PCTH_avg_IVFPQ_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    # np.save(topk_idx_path + 'topk_idx_img_PCTH_avg_IVFPQ.npy', topk_idx_img)
    # np.save(topk_idx_path + 'topk_idx_que_PCTH_avg_IVFPQ.npy', topk_idx_que)
    #
    # topk_idx_img = get_topk_triplets_idx(PCTH_avg_IVFPQ_pca_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    # topk_idx_que = get_topk_triplets_idx(PCTH_avg_IVFPQ_pca_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    # np.save(topk_idx_path + 'topk_idx_img_PCTH_avg_IVFPQ_pca.npy', topk_idx_img)
    # np.save(topk_idx_path + 'topk_idx_que_PCTH_avg_IVFPQ_pca.npy', topk_idx_que)

    # load doc index
    PC_avg_HNSW_index = faiss.read_index(index_path + 'PC_avg_HNSW.index')
    PC_avg_HNSW_pca_index = faiss.read_index(index_path + 'PC_avg_HNSW_pca.index')
    PCTH_avg_HNSW_index = faiss.read_index(index_path + 'PCTH_avg_HNSW.index')
    PCTH_avg_HNSW_pca_index = faiss.read_index(index_path + 'PCTH_avg_HNSW_pca.index')
    print('-- get index')
    # search
    topk_idx_img = get_topk_triplets_idx(PC_avg_HNSW_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    topk_idx_que = get_topk_triplets_idx(PC_avg_HNSW_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    np.save(topk_idx_path + 'topk_idx_img_PC_avg_HNSW.npy', topk_idx_img)
    np.save(topk_idx_path + 'topk_idx_que_PC_avg_HNSW.npy', topk_idx_que)

    topk_idx_img = get_topk_triplets_idx(PC_avg_HNSW_pca_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    topk_idx_que = get_topk_triplets_idx(PC_avg_HNSW_pca_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    np.save(topk_idx_path + 'topk_idx_img_PC_avg_HNSW_pca.npy', topk_idx_img)
    np.save(topk_idx_path + 'topk_idx_que_PC_avg_HNSW_pca.npy', topk_idx_que)

    topk_idx_img = get_topk_triplets_idx(PCTH_avg_HNSW_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    topk_idx_que = get_topk_triplets_idx(PCTH_avg_HNSW_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    np.save(topk_idx_path + 'topk_idx_img_PCTH_avg_HNSW.npy', topk_idx_img)
    np.save(topk_idx_path + 'topk_idx_que_PCTH_avg_HNSW.npy', topk_idx_que)

    topk_idx_img = get_topk_triplets_idx(PCTH_avg_HNSW_pca_index, img_tokens_xq, img_xq.shape[1])  # (bs, k)
    topk_idx_que = get_topk_triplets_idx(PCTH_avg_HNSW_pca_index, que_tokens_xq, que_xq.shape[1])  # (bs, k)
    np.save(topk_idx_path + 'topk_idx_img_PCTH_avg_HNSW_pca.npy', topk_idx_img)
    np.save(topk_idx_path + 'topk_idx_que_PCTH_avg_HNSW_pca.npy', topk_idx_que)