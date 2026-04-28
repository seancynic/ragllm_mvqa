import datasets
import torch
from torch.utils.data import Dataset
from .transform import image_transform


class UMLSRELDefDataset(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int):
        self.dataset = datasets.Dataset.from_pandas(dataset)
        self.tokenizer = tokenizer
        self.txt_length = txt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        kg = self.dataset[idx]
        head_ids, head_am = self._get_token_ids(kg['CUI1'])
        pos_tail_ids, pos_tail_am = self._get_token_ids(kg['CUI2'])
        neg_tail_ids, neg_tail_am = self._get_token_ids(kg['neg_CUI2'])
        return {'head_ids': head_ids, 'head_am': head_am,
                'rel_id': kg['REL'],
                'pos_tail_ids': pos_tail_ids, 'pos_tail_am': pos_tail_am,
                'neg_tail_ids': neg_tail_ids, 'neg_tail_am': neg_tail_am
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class PMCVQADatasetRaw(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int, stage: str):
        self.dataset = datasets.Dataset.from_pandas(dataset).cast_column('Figure_path', datasets.Image())
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # form image
        image = self.image2tensor(item['Figure_path'])
        # form question
        que_ids, que_am = self._get_token_ids(item['Question'].strip())  # (bs, 77)
        # form answer
        answer = item['Answer'].strip()
        ans_ids, _ = self._get_token_ids(answer)  # (bs, 77)
        # form label
        ans_ids[ans_ids == 0] = -100

        return {'img_pt': image,
                'que_ids': que_ids, 'que_am': que_am,  # (bs, 77)
                'ans_ids': ans_ids, 'ans_txt': answer  # (bs, 77)
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class PMCVQADatasetSEP(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int, stage: str):
        self.dataset = datasets.Dataset.from_pandas(dataset).cast_column('Figure_path', datasets.Image())
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # form image
        image = self.image2tensor(item['Figure_path'])
        # form prompt
        prompt_before_ids, prompt_before_am = torch.tensor([2]), torch.tensor([1])
        prompt_after_ids, prompt_after_am = self._get_token_ids(item['Question'].strip())  # (bs, 77)
        prompt_after_ids[0] = 3
        # form answer
        answer = item['Answer'].strip()
        ans_ids, _ = self._get_token_ids(answer)  # (bs, 77)
        # generate labels
        ans_ids[ans_ids == 0] = -100

        return {'img_pt': image,
                'prompt_before_ids': prompt_before_ids, 'prompt_before_am': prompt_before_am,  # (bs, 1)
                'prompt_after_ids': prompt_after_ids, 'prompt_after_am': prompt_after_am,  # (bs, 77)
                'ans_ids': ans_ids, 'ans_txt': answer,  # (bs, 77)
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class PMCVQADatasetTriHolder(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int, stage: str):
        self.dataset = datasets.Dataset.from_pandas(dataset).cast_column('Figure_path', datasets.Image())
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # form image
        image = self.image2tensor(item['Figure_path'])
        # form prompt
        prompt_before_ids, prompt_before_am = torch.tensor([2, 0, 0, 0, 3]), torch.tensor([1, 0, 0, 0, 1])
        prompt_after_ids, prompt_after_am = self._get_token_ids(item['Question'].strip())  # (bs, 77)
        prompt_after_ids[0] = 3
        # form answer
        answer = item['Answer'].strip()
        ans_ids, _ = self._get_token_ids(answer)  # (bs, 77)
        # generate labels
        ans_ids[ans_ids == 0] = -100

        return {'img_pt': image,
                'prompt_before_ids': prompt_before_ids, 'prompt_before_am': prompt_before_am,  # (bs, 5)
                'prompt_after_ids': prompt_after_ids, 'prompt_after_am': prompt_after_am,  # (bs, 77)
                'ans_ids': ans_ids, 'ans_txt': answer,  # (bs, 77)
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class PMCVQADatasetWordSEP(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int, stage: str):
        self.dataset = datasets.Dataset.from_pandas(dataset).cast_column('Figure_path', datasets.Image())
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # form image
        image = self.image2tensor(item['Figure_path'])
        # form prompt
        # prompt_before = '<image>'
        # prompt_after = '</image><question>' + item['Question'].strip() + '</question><answer>'
        prompt_before = 'image'
        prompt_after = 'question ' + item['Question'].strip()
        prompt_before_ids, prompt_before_am = self._get_token_ids(prompt_before, 3)
        prompt_after_ids, prompt_after_am = self._get_token_ids(prompt_after, self.txt_length)
        # form answer
        # answer = item['Answer'].strip() + '</answer>'
        answer = item['Answer'].strip()
        ans_ids, _ = self._get_token_ids(answer, self.txt_length)
        # form label
        ans_ids[ans_ids == 0] = -100

        # return {'img_pt': image, 'img_am': torch.ones(50),
        #         'prompt_before_ids': prompt_before_ids[:4], 'prompt_before_am': prompt_before_am[:4],
        #         'prompt_after_ids': prompt_after_ids[1:], 'prompt_after_am': prompt_after_am[1:],
        #         'ans_ids': ans_ids, 'ans_txt': answer
        #         }
        return {'img_pt': image,
                'prompt_before_ids': prompt_before_ids[:2], 'prompt_before_am': prompt_before_am[:2],  # (bs, 2)
                'prompt_after_ids': prompt_after_ids[1:], 'prompt_after_am': prompt_after_am[1:],  # (bs, 76)
                'ans_ids': ans_ids, 'ans_txt': answer
                }

    def _get_token_ids(self, text, txt_len):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=txt_len,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class PMCVQADatasetRawTri(Dataset):
    def __init__(self, dataset, triplets, tokenizer, txt_length: int, stage: str):
        self.dataset = datasets.Dataset.from_pandas(dataset).cast_column('Figure_path', datasets.Image())
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # form image
        image = self.image2tensor(item['Figure_path'])
        # form question
        que_ids, que_am = self._get_token_ids(item['Question'].strip())  # (bs, 77)
        # form answer
        answer = item['Answer'].strip()
        ans_ids, _ = self._get_token_ids(answer)  # (bs, 77)
        # form label
        ans_ids[ans_ids == 0] = -100

        return {'img_pt': image,
                'que_ids': que_ids, 'que_am': que_am,  # (bs, 77)
                'ans_ids': ans_ids, 'ans_txt': answer  # (bs, 77)
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class VQARADDatasetHTML(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int, stage: str):
        if stage == 'train':
            self.dataset = datasets.load_dataset('flaviagiammarino/vqa-rad', split='train')
        else:
            self.dataset = datasets.load_dataset('flaviagiammarino/vqa-rad', split='test')
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # form image
        image = self.image2tensor(item['Figure_path'])
        # form prompt
        prompt_before = '<image>'
        prompt_after = '</image><question>' + item['question'].strip() + '</question><answer>'
        prompt_before_ids, prompt_before_am = self._get_token_ids(prompt_before)
        prompt_after_ids, prompt_after_am = self._get_token_ids(prompt_after)
        # form answer
        answer = item['answer'].strip() + '</answer>'
        ans_ids, _ = self._get_token_ids(answer)
        ans_ids[ans_ids == 0] = -100

        return {'img_pt': image, 'img_am': torch.ones(50),
                'prompt_before_ids': prompt_before_ids[:4], 'prompt_before_am': prompt_before_am[:4],
                'prompt_after_ids': prompt_after_ids[1:], 'prompt_after_am': prompt_after_am[1:],
                'ans_ids': ans_ids, 'ans_txt': answer
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)


class PathVQADatasetHTML(Dataset):
    def __init__(self, dataset, tokenizer, txt_length: int, stage: str):
        if stage == 'train':
            self.dataset = datasets.load_dataset('flaviagiammarino/path-vqa', split='train')
        elif stage == 'valid':
            self.dataset = datasets.load_dataset('flaviagiammarino/path-vqa', split='validation')
        else:
            self.dataset = datasets.load_dataset('flaviagiammarino/path-vqa', split='test')
        self.tokenizer = tokenizer
        self.txt_length = txt_length
        self.image2tensor = image_transform(image_size=224)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # form image
        image = self.image2tensor(item['Figure_path'])
        # form prompt
        prompt_before = '<image>'
        prompt_after = '</image><question>' + item['question'].strip() + '</question><answer>'
        prompt_before_ids, prompt_before_am = self._get_token_ids(prompt_before)
        prompt_after_ids, prompt_after_am = self._get_token_ids(prompt_after)
        # form answer
        answer = item['answer'].strip() + '</answer>'
        ans_ids, _ = self._get_token_ids(answer)
        ans_ids[ans_ids == 0] = -100

        return {'img_pt': image, 'img_am': torch.ones(50),
                'prompt_before_ids': prompt_before_ids[:4], 'prompt_before_am': prompt_before_am[:4],
                'prompt_after_ids': prompt_after_ids[1:], 'prompt_after_am': prompt_after_am[1:],
                'ans_ids': ans_ids, 'ans_txt': answer
                }

    def _get_token_ids(self, text):
        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.txt_length,
                                return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0]  # (txt_length)
