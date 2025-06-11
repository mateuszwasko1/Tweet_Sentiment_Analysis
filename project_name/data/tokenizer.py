from typing import Sequence, Tuple
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    """
    PyTorch Dataset for a collection of tweets and their labels.

    Attributes:
        tweets (list[str]): List of tweet texts.
        labels (list[int]): Corresponding integer labels for each tweet.
    """

    def __init__(self, tweets: Sequence[str], labels: Sequence[int]) -> None:
        """
        Initialize the TweetDataset with tweets and labels.

        Args:
            tweets (Sequence[str]): Iterable of tweet texts.
            labels (Sequence[int]): Iterable of integer labels
            corresponding to each tweet.

        Raises:
            ValueError: If the number of tweets and labels do not match.
        """
        tweets_list: list[str] = list(tweets)
        labels_list: list[int] = list(labels)
        if len(tweets_list) != len(labels_list):
            raise ValueError(
                f"Tweets and labels must have the same length; "
                f"got {len(tweets_list)} tweets and {len(labels_list)} labels."
            )
        self.tweets: list[str] = tweets_list
        self.labels: list[int] = labels_list

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

        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self.tweets):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}."
            )
        return self.tweets[idx], self.labels[idx]
