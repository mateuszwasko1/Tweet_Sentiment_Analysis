from torch.utils.data import Dataset
import torch


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id=None,
                 max_length=128):
        self.texts = texts.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build label2id mapping if not provided
        if label2id is None:
            unique = sorted(set(labels))
            self.label2id = {label: i for i, label in enumerate(unique)}
        else:
            self.label2id = label2id

        self.labels = [self.label2id[label] for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize single tweet (no batch!)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_token_type_ids=False,
            return_tensors="pt"
        )

        # Remove batch dim and add label
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item
