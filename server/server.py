# server.py

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.app import start_server

def aggregate_metrics(metrics_list):
    """Aggregate client metrics (e.g., average accuracy)."""
    num_clients = len(metrics_list)
    accuracy_sum = sum(m[1]["accuracy"] for m in metrics_list)
    return {"accuracy": accuracy_sum / num_clients}

def get_strategy() -> fl.server.strategy.FedAvg:
    """Build federated averaging strategy with custom settings."""
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

def main():
    print("[Server] Initializing...")
    # Strategy and server config
    strategy = get_strategy()
    config = ServerConfig(num_rounds=3)

    # Start the federated learning server
    start_server(
        server_address="192.168.1.208:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutdown requested. Exiting gracefully.")
