from torch.utils.data import Dataset
from typing import Sequence, Tuple, List


class TweetDataset(Dataset):
    """
    PyTorch Dataset for a collection of tweets and their labels.

    Attributes:
        tweets (List[str]): List of tweet texts.
        labels (List[int]): Corresponding integer labels for each tweet.
    """
    def __init__(self, tweets: Sequence[str], labels: Sequence[int]) -> None:
        """
        Initialize the TweetDataset with tweets and labels.

        Args:
            tweets (Sequence[str]): Iterable of tweet texts.
            labels (Sequence[int]): Iterable of integer labels
            corresponding to each tweet.
        """
        self.tweets: List[str] = list(tweets)
        self.labels: List[int] = list(labels)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of tweets in the dataset.
        """
        return len(self.tweets)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Retrieve the tweet text and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[str, int]: A tuple of the tweet text and its integer label.
        """
        return self.tweets[idx], self.labels[idx]
