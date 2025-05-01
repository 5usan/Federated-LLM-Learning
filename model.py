# model.py
import torch
import torch.nn as nn
from transformers import DistilBertModel
from fairscale.nn import Pipe

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# def load_model(num_labels: int = 3) -> DistilBertForSequenceClassification:
#     """
#     Loads the DistilBERT model for sequence classification.

#     Args:
#         num_labels (int): Number of output classes. Defaults to 2 (binary classification).

#     Returns:
#         model (DistilBertForSequenceClassification): The loaded model.
#     """
# model = DistilBertForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased",
# num_labels=num_labels,
# ignore_mismatched_sizes=True,
# output_attentions=False,
# output_hidden_states=False
# )
#     return model

# ----- WRAPPER FOR DISTILBERT ENCODER -----
class DistilBertWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3,
            ignore_mismatched_sizes=True,
            output_attentions=False,
            output_hidden_states=False,
        )

    def forward(self, inputs):
        input_ids, attention_mask, labels = inputs
        # Pass inputs to HuggingFace model using keyword arguments
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Return pooled output and labels to next stage
        return (pooled_output, labels)


# ----- WRAPPER FOR CLASSIFIER -----
class ClassifierWrapper(nn.Module):
    def __init__(self, hidden_size=768, num_labels=3):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs):
        pooled_output, labels = inputs
        logits = self.classifier(pooled_output)
        return (logits, labels)


# ----- LOAD MODEL FUNCTION FOR CLIENT -----
def load_model(num_labels=3):
    # Stack layers into a sequential model
    model = nn.Sequential(DistilBertWrapper(), ClassifierWrapper(num_labels=num_labels))

    # Create pipelined model across devices
    piped_model = Pipe(
        model,
        balance=[1, 1],  # 1 layer per device
        devices=["cuda:0", "cpu"],  # adjust based on your setup
        chunks=8,  # number of microbatches
    )

    return piped_model


def load_tokenizer() -> DistilBertTokenizer:
    """
    Loads the DistilBERT tokenizer.

    Returns:
        tokenizer (DistilBertTokenizer): The loaded tokenizer.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer
