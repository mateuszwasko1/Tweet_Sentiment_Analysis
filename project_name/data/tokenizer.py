from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, tweets, labels):
        self.tweets = tweets.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]
