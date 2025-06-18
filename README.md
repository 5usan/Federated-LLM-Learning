# Federated-LLM-Learning

A research project to train **DistilBERT** for sentiment classification using **Federated Learning** with the **Flower** framework. Each client trains locally on its own data; updates are aggregated on the server. Model parallelism is supported via **FairScale**'s `Pipe`, enabling large model training across CPU/GPU.

---

## 🔧 Installation

1. **Clone repo**:
   ```bash
   git clone https://github.com/5usan/Federated-LLM-Learning.git
   cd Federated-LLM-Learning
   ```

2. **Set up environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt  # or install manually (see below)
   ```

3. **Required Libraries**:
   - `torch`, `transformers`, `flwr`, `fairscale`, `scikit-learn`, `pandas`, `datasets`

---

## 🗂️ Project Structure

```
├── client/             # Flower client code
├── server/             # Flower server code
├── data/               # Data loading & CSVs
├── model.py            # DistilBERT + classifier with pipeline support
├── utlis.py            # Dataset wrappers (note: "utils" typo)
├── centralized.ipynb   # Central training baseline
```

---

## 🚀 Running the Project

1. **Start Server**:
   ```bash
   python server/server.py
   ```

2. **Start Clients** (in separate terminals):
   ```bash
   python -m client.client 0
   python -m client.client 1
   python -m client.client 2
   ```

> Each client will load its data split and train locally. The server aggregates updates via FedAvg.

---

## 🧪 Dataset

- Uses **Twitter Financial News Sentiment** dataset (`data/train.csv`, `data/test.csv`)
- Format: CSV with `text` and `label` columns
- Optional: use IMDb dataset via HuggingFace `datasets`

---

## 📈 Features

- Federated text classification with DistilBERT  
- Pipeline parallelism (model split across devices)  
- Evaluation: Accuracy, Precision, Recall, F1  
- Centralized training notebook for comparison

---

## 📜 License & Citation

- **License**: MIT
- **Citation**:
```bibtex
@misc{FederatedLLMLearning2025,
  author = {Susan Shrestha},
  title = {Federated-LLM-Learning: Federated Learning for LLMs},
  year = {2025},
  howpublished = {\url{https://github.com/5usan/Federated-LLM-Learning}},
}
```