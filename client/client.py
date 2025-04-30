# client.py

import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import load_model
from utlis import LocalTextDataset
import sys
from data.dataloader import load_twitter_partition 


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
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(1):  # 1 local epoch per round
            print(f"[Client] Training on epoch {epoch + 1}")
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("[Client] Starting evaluation")
        self.set_parameters(parameters)
        self.model.eval()

        correct, total = 0, 0
        loss_total = 0.0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                preds = torch.argmax(outputs.logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loss_total += loss.item()

        if total == 0:
            print("[Client] Warning: test set is empty!")
            return 0.0, 0, {}

        avg_loss = loss_total / total
        accuracy = correct / total

        print(f"[Client] Eval completed: loss={avg_loss:.4f}, acc={accuracy:.4f}")
        return avg_loss, total, {"accuracy": accuracy}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client] Using device: {device}")
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    train_texts, train_labels, test_texts, test_labels = load_twitter_partition(client_id, num_clients=5)
    train_dataset = LocalTextDataset(train_texts, train_labels)
    test_dataset = LocalTextDataset(test_texts, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    print(f"[Client {client_id}] Loaded data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    model = load_model()
    model.to(device)

    client = FlowerClient(model, train_loader, test_loader, device)
    fl.client.start_numpy_client(server_address="192.168.1.208:8080", client=client)


if __name__ == "__main__":
    main()
