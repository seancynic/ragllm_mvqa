import datasets
from torch.utils.data import Dataset
from .transform import image_transform


class QueryDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = datasets.Dataset.from_pandas(dataset).cast_column('Figure_path', datasets.Image())
        self.image2tensor = image_transform(image_size=224, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.image2tensor(item['Figure_path'])
        question = item['Question'].strip()

        return {
            'img_pt': image,
            'que_txt': question,
        }
