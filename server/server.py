# server.py

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.app import start_server


def aggregate_metrics(metrics_list):
    num_clients = len(metrics_list)
    metrics_sum = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for _, metrics in metrics_list:
        for key in metrics_sum:
            metrics_sum[key] += metrics.get(key, 0.0)
    averaged = {key: metrics_sum[key] / num_clients for key in metrics_sum}
    print(f"[Server] Aggregated Metrics: {averaged}")
    return averaged


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


port = 8080


def main():
    print(f"[Server] Server initializing at {port}. Waiting for clients...")
    # Strategy and server config
    strategy = get_strategy()
    config = ServerConfig(num_rounds=1)

    # Start the federated learning server
    start_server(
        server_address=f"192.168.1.208:{port}",
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutdown requested. Exiting gracefully.")
