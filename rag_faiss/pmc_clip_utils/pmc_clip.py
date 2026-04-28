import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

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
        self.tokenizer = AutoTokenizer.from_pretrained(text_cfg['bert_model_name'])

    def get_token_ids(self, text):
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            add_special_tokens=False,
            return_tensors="pt"
        )

        return encoded["input_ids"], encoded["attention_mask"]

    def encode_text_embedding(self, input_ids, am):
        return self.text_encoder(input_ids=input_ids, attention_mask=am, output_attentions=False)['hidden_states'][0]

    def encode_text_cls(self, input_ids, am):
        return self.text_encoder(input_ids=input_ids, attention_mask=am, output_attentions=False)['pooler_output']

    def encode_text_all(self, input_ids, am):
        return self.text_encoder(input_ids=input_ids, attention_mask=am, output_attentions=False)['last_hidden_state']

    def encode_image(self, image):
        return self.visual(image)

    def _get_input_embeds(self, image, question):
        # image
        img_reps = self.encode_image(image).permute(1, 0, 2)  # (bs, 3, 224, 224) -> (bs, 50, 768)
        img_reps = F.normalize(img_reps, dim=-1)
        # question
        question_reps = self.encode_text_all(*self.get_token_ids(question))  # (bs, 77, 768)
        question_reps = F.normalize(question_reps, dim=-1)
        # image + question
        return torch.cat((img_reps, question_reps), 1)  # (bs, 50 + 77, 768)

    @staticmethod
    def _get_key_padding_mask(token_ids, type: int):
        if type == 0:
            key_padding_mask = torch.zeros(token_ids.size())
            key_padding_mask[token_ids == 0] = -torch.inf
            return key_padding_mask
        elif type == 1:
            key_padding_mask = torch.ones(token_ids.size())
            key_padding_mask[token_ids == 0] = 0
            return key_padding_mask
        else:
            raise RuntimeError('Wrong Padding Mask Type!')


    def forward(self, batch, stage):
        if stage == 'decoder_tvt':
            # generate memory
            memory = self._get_input_embeds(batch['img_pt'], batch['que_txt'])

            # generate answer ids
            answer_ids, am = self.get_token_ids(batch['ans_txt'])  # (bs, 77)
            input_ids, label = answer_ids[:, :-1], answer_ids[:, 1:]  # no last token (input_ids), no cls (label)

            # generate answer embeddings (tgt)
            answer_embeds = self.encode_text_embedding(input_ids, am)
            answer_embeds = F.normalize(answer_embeds, dim=-1)

            # generate tgt key padding mask
            tgt_key_padding_mask = self._get_key_padding_mask(input_ids, 0).to(self.device)

            # generate tgt mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(answer_embeds.size(1)).to(self.device)

            return {
                'memory': memory,  # (bs, 50 + 77, 768)
                'label': label,  # (bs, 76)
                'tgt': answer_embeds,  # (bs, 76, 768)
                'tgt_key_padding_mask': tgt_key_padding_mask,  # (bs, 76)
                'tgt_mask': tgt_mask  # (76, 76)
            }

        elif stage == 'decoder_cls':
            # generate memory
            memory = self._get_input_embeds(batch['img_pt'], batch['que_txt'])

            # generate answer ids
            answer_ids, am = self.get_token_ids(batch['ans_txt'])
            input_ids = answer_ids[:, :1]  # only cls (input_ids)

            # generate answer embeddings (tgt)
            answer_embeds = self.encode_text_embedding(input_ids, am)
            answer_embeds = F.normalize(answer_embeds, dim=-1)

            # generate tgt key padding mask
            tgt_key_padding_mask = self._get_key_padding_mask(input_ids, 0).to(self.device)

            # generate tgt mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(answer_embeds.size(1)).to(self.device)

            return {
                'memory': memory,  # (bs, 50 + 77, 768)
                'first_input_ids': input_ids,  # (bs, 1)
                'tgt': answer_embeds,  # (bs, 1, 768)
                'tgt_key_padding_mask': tgt_key_padding_mask,  # (bs, 1)
                'tgt_mask': tgt_mask  # (1, 1)
            }

        elif stage == 'decoder_pred':
            # generate answer embeddings (tgt)
            answer_embeds = self.encode_text_embedding(batch['next_input_ids'])
            answer_embeds = F.normalize(answer_embeds, dim=-1)

            # generate tgt key padding mask
            tgt_key_padding_mask = self._get_key_padding_mask(batch['next_input_ids'], 0).to(self.device)

            # generate tgt mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(answer_embeds.size(1)).to(self.device)

            return {
                'tgt': answer_embeds,  # (bs, t + 1, 768)
                'tgt_key_padding_mask': tgt_key_padding_mask,  # (bs, t + 1)
                'tgt_mask': tgt_mask  # (t + 1, t + 1)
            }

        elif stage == 'llm_raw':
            # image
            img_embeds = self.encode_image(batch['img_pt']).permute(1, 0, 2)  # (bs, 3, 224, 224) -> (bs, 50, 768)
            img_embeds =  F.normalize(img_embeds, dim=-1)  # (bs, 50, 768)

            # question
            que_embeds = self.encode_text_all(batch['que_ids'], batch['que_am'])  # (bs, 77, 768)
            que_embeds = F.normalize(que_embeds, dim=-1)  # (bs, 77, 768)

            # concat
            inputs_embeds = torch.cat((img_embeds, que_embeds), dim=1)  # (bs, 50 + 77, 768)

            return inputs_embeds

        elif stage == 'llm_prompt':
            # image
            img_embeds = self.encode_image(batch['img_pt']).permute(1, 0, 2)  # (bs, 3, 224, 224) -> (bs, 50, 768)
            img_embeds = F.normalize(img_embeds, dim=-1)  # (bs, 50, 768)

            # prompt
            prompt_before_embeds = self.encode_text_all(batch['prompt_before_ids'], batch['prompt_before_am'])  # (bs, ?, 768)
            prompt_before_embeds = F.normalize(prompt_before_embeds, dim=-1)  # (bs, ?, 768)
            prompt_after_embeds = self.encode_text_all(batch['prompt_after_ids'], batch['prompt_after_am'])  # (bs, ?, 768)
            prompt_after_embeds = F.normalize(prompt_after_embeds, dim=-1)  # (bs, ?, 768)

            # concat
            inputs_embeds = torch.cat((prompt_before_embeds, img_embeds, prompt_after_embeds), dim=1)  # (bs, ? + 50 + ?, 768)

            return inputs_embeds

        elif stage == 'img_que_mat':
            img_mats = self.encode_image(batch['img_pt'].to('cuda')).permute(1, 0, 2)  # (bs, 3, 224, 224) -> (bs, 50, 768)
            que_mats = self.encode_text_all(batch['que_ids'].to('cuda'), batch['que_am'].to('cuda'))  # (bs, 77, 768)

            return {
                'image_mats': img_mats,
                'question_mats': que_mats
            }