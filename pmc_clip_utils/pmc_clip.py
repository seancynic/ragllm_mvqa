import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .blocks import ModifiedResNet


class PMC_CLIP(nn.Module):
    def __init__(self, embed_dim: int, vision_cfg, text_cfg):
        super().__init__()
        # Image
        self.visual = ModifiedResNet(
            layers=vision_cfg['layers'],
            output_dim=embed_dim,
            heads=vision_cfg['width'] * 32 // vision_cfg['head_width'],
            image_size=vision_cfg['image_size'],
            width=vision_cfg['width']
        )
        # Text
        self.text_encoder = AutoModel.from_pretrained(text_cfg['bert_model_name'], output_hidden_states=True)

    def _encode_text_embedding(self, input_ids, am):
        return self.text_encoder(
            input_ids=input_ids.to('cuda'),
            attention_mask=am.to('cuda'),
            output_attentions=False
        )['hidden_states'][0]

    def _encode_text_cls(self, input_ids, am):
        return self.text_encoder(
            input_ids=input_ids.to('cuda'),
            attention_mask=am.to('cuda'),
            output_attentions=False
        )['pooler_output']

    def _encode_text_all(self, input_ids, am):
        return self.text_encoder(
            input_ids=input_ids.to('cuda'),
            attention_mask=am.to('cuda'),
            output_attentions=False
        )['last_hidden_state']

    def _encode_image(self, image):
        return self.visual(image.to('cuda')).permute(1, 0, 2)

    def forward(self, batch, stage):
        if stage == 'query_mean':
            # Encode images and questions
            img_embed = self._encode_image(batch['img_pt']).mean(dim=1, keepdim=True)
            que_embed = self._encode_text_all(batch['que_ids'], batch['que_am']).mean(dim=1, keepdim=True)

            return {
                'images_embeddings': img_embed.detach().cpu(),  # (bs, 1, 768)
                'questions_embeddings': que_embed.detach().cpu()  # (bs, 1, 768)
            }

        elif stage == 'query':
            # Encode images and questions
            img_embed = self._encode_image(batch['img_pt'])  # (bs, 3, 224, 224) -> (bs, 50, 768)
            que_embed = self._encode_text_all(batch['que_ids'], batch['que_am'])  # (bs, que_len, 768)

            return {
                'images_embeddings': img_embed.detach().cpu(),
                'questions_embeddings': que_embed.detach().cpu()
            }

        elif stage == 'document_mean':
            # Encode documents
            doc_embed = self._encode_text_all(batch['doc_ids'], batch['doc_am'])  # (bs, doc_len, 768)
            # Remove padding, normalize and form a list
            doc_embeddings = [
                doc_embed[i, mask.to(dtype=torch.bool)].mean(dim=0, keepdim=True).detach().cpu()  # (1, 768)
                for i, mask in enumerate(batch['doc_am'])
            ]

            return doc_embeddings  # [(1, 768)]

        elif stage == 'document_mean_norm':
            # Encode documents
            doc_embed = self._encode_text_all(batch['doc_ids'], batch['doc_am'])  # (bs, doc_len, 768)
            # Remove padding, normalize and form a list
            doc_embeddings = [
                F.normalize(doc_embed[i, mask.to(dtype=torch.bool)].mean(dim=0, keepdim=True), dim=-1).detach().cpu()  # (1, 768)
                for i, mask in enumerate(batch['doc_am'])
            ]

            return doc_embeddings  # [(1, 768)]

        elif stage == 'document':
            # Encode documents
            doc_embed = self._encode_text_all(batch['doc_ids'], batch['doc_am'])  # (bs, doc_len, 768)
            # Remove padding, normalize and form a list
            doc_embeddings = [
                doc_embed[i, mask.to(dtype=torch.bool)].detach().cpu()
                for i, mask in enumerate(batch['doc_am'])
            ]

            return doc_embeddings  # [(n, 768)]

        elif stage == 'document_norm':
            # Encode documents
            doc_embed = self._encode_text_all(batch['doc_ids'], batch['doc_am'])  # (bs, doc_len, 768)
            # Remove padding, normalize and form a list
            doc_embeddings = [
                F.normalize(doc_embed[i, mask.to(dtype=torch.bool)], dim=-1).detach().cpu()
                for i, mask in enumerate(batch['doc_am'])
            ]

            return doc_embeddings  # [(n, 768)]

        else:
            print(f'ERROR: Unknown stage ({stage})!')
