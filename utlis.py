# utils.py

import torch
from torch.utils.data import Dataset
from typing import List
from model import load_tokenizer  # make sure this is defined in model.py

class LocalTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], max_length: int = 128):
        self.tokenizer = load_tokenizer()
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encodings = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
