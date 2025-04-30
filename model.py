# model.py

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def load_model(num_labels: int = 2) -> DistilBertForSequenceClassification:
    """
    Loads the DistilBERT model for sequence classification.
    
    Args:
        num_labels (int): Number of output classes. Defaults to 2 (binary classification).
        
    Returns:
        model (DistilBertForSequenceClassification): The loaded model.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    return model

def load_tokenizer() -> DistilBertTokenizer:
    """
    Loads the DistilBERT tokenizer.
    
    Returns:
        tokenizer (DistilBertTokenizer): The loaded tokenizer.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer
