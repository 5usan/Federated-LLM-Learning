import torch
import torch.nn as nn
from fairscale.nn.pipe import Pipe
from transformers import DistilBertModel, DistilBertTokenizer

# ✅ Step 1: Define DistilBERT blocks (split manually)
class DistilBertBlock1(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        model = DistilBertModel.from_pretrained(model_name)
        # Take embedding and first 3 transformer blocks
        self.embeddings = model.embeddings
        self.transformer_blocks = nn.Sequential(*model.transformer.layer[:3])

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        for layer in self.transformer_blocks:
            x = layer(x, attention_mask=attention_mask)[0]
        return x


class DistilBertBlock2(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        model = DistilBertModel.from_pretrained(model_name)
        self.transformer_blocks = nn.Sequential(*model.transformer.layer[3:])
        self.pre_classifier = nn.Linear(model.config.dim, model.config.dim)
        self.classifier = nn.Linear(model.config.dim, num_labels)
        self.dropout = nn.Dropout(model.config.seq_classif_dropout)

    def forward(self, hidden_state, attention_mask):
        for layer in self.transformer_blocks:
            hidden_state = layer(hidden_state, attention_mask=attention_mask)[0]
        pooled_output = hidden_state[:, 0]
        x = self.dropout(self.pre_classifier(pooled_output))
        return self.classifier(x)

# ✅ Step 2: Wrap blocks into torch.nn.Sequential for Pipe
class DistilBertPipeline(nn.Sequential):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        block1 = DistilBertBlock1(model_name)
        block2 = DistilBertBlock2(model_name, num_labels)
        # Wrap each block in lambda so they match Pipe input/output
        super().__init__(
            lambda x: (block1(x[0], x[1]), x[1]),     # (hidden_state, attention_mask)
            lambda x: block2(x[0], x[1])              # output logits
        )

# ✅ Step 3: Run the pipeline
def run_pipeline():
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    pipe_model = DistilBertPipeline()

    # Split across GPUs with Pipe
    pipe = Pipe(pipe_model, devices=[device0, device1], chunks=2)

    # Dummy input
    text = "I love pipeline parallelism in PyTorch"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = inputs["input_ids"].to(device0)
    attention_mask = inputs["attention_mask"].to(device0)

    # Forward pass
    with torch.no_grad():
        output = pipe((input_ids, attention_mask))
        print("Logits:", output)

if __name__ == "__main__":
    run_pipeline()
