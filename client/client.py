# client.py

import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import load_model
from utlis import LocalTextDataset
import sys
from data.dataloader import load_twitter_partition 
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        # training start time
        training_start_time = time.time()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
        for epoch in range(10):
            print(f"[Client] Training on epoch {epoch + 1}")
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits, labels = self.model((input_ids, attention_mask, labels))
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        training_finish_time = time.time()
        print(f"Training ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_finish_time))}")
        training_time = training_finish_time - training_start_time
        print(f"[Client] Training completed in {training_time:.2f} seconds")

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("[Client] Starting evaluation")
        self.set_parameters(parameters)
        self.model.eval()

        y_true, y_pred = [], []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits, labels = self.model((input_ids, attention_mask, labels))
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                num_batches += 1
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"[Client] Evaluation Results - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return avg_loss, len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client] Using device: {device}")
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    train_texts, train_labels, test_texts, test_labels = load_twitter_partition(client_id, num_clients=3)
    train_dataset = LocalTextDataset(train_texts, train_labels)
    test_dataset = LocalTextDataset(test_texts, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
  
    print(f"[Client {client_id}] Loaded data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    model = load_model()
    # model.to(device)

    client = FlowerClient(model, train_loader, test_loader, device)
    fl.client.start_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
    main()
