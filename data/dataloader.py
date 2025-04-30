# data_loader.py

import os
from datasets import load_dataset
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
import pandas as pd
TWITTER_PREPROCESSED_DATA = os.path.join(
    os.getcwd(), "data", "TwitterPreprocessedDataset.csv"
)

def load_imdb_partition(client_id: int, num_clients: int = 5) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load a partition of the IMDb dataset for a given client.

    Args:
        client_id (int): Index of the client (0-based).
        num_clients (int): Total number of clients.

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    dataset = load_dataset("imdb")

    # Split training data among clients (non-overlapping)
    full_train = dataset["train"]
    total = len(full_train)
    shard_size = total // num_clients

    start = client_id * shard_size
    end = start + shard_size
    print(f"[Client {client_id}] Loading data from {start} to {end}")
    train_split = full_train.select(range(start, end))

    # Use same test set for all clients (or subsample here too)
    test_split = dataset["test"].select(range(100))  # uniform test size

    return train_split["text"], train_split["label"], test_split["text"], test_split["label"]


def load_twitter_partition(client_id: int, num_clients: int = 5) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load a partition of the Twitter dataset for a given client.

    Args:
        client_id (int): Index of the client (0-based).
        num_clients (int): Total number of clients.

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    df = pd.read_csv(TWITTER_PREPROCESSED_DATA)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset

    # Split data among clients (non-overlapping)
    total = len(df)
    shard_size = total // num_clients

    start = client_id * shard_size
    end = start + shard_size
    print(f"[Client {client_id}] Loading data from {start} to {end}")
    train_split = df.iloc[start:end]

    # Use same test set for all clients (or subsample here too)
    test_split = df.iloc[:100]  # uniform test size

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_split["label"])
    test_labels = label_encoder.transform(test_split["label"])

    return train_split["feature"].tolist(), train_labels.tolist(), test_split["feature"].tolist(), test_labels.tolist()
