import flwr as fl
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Dict

# --- 0. Dependency Check ---
# This script requires PyTorch and Opacus. Install with: pip install torch opacus
from opacus import PrivacyEngine

# --- 1. The PyTorch Model ---
class TorchLogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

# --- 2. Data Preparation (Run Once) ---
def prepare_federated_data():
    client_data = {}
    all_feat_cols = set()

    for hospital_id in ['A', 'B', 'C']:
        data_path = Path("/Users/faezehhosseini/phd_federated_learning_project/data/processed_data")
        X = pd.read_csv(data_path / f"X_{hospital_id}.csv")
        y = pd.read_csv(data_path / f"y_{hospital_id}.csv")
        client_data[hospital_id] = (X, y)
        all_feat_cols.update(X.columns)
    
    all_feat_cols = sorted(list(all_feat_cols))
    print(f"Total unique features across all clients: {len(all_feat_cols)}")

    total_n = 0
    sum_vec = np.zeros(len(all_feat_cols), dtype=np.float64)
    sumsq_vec = np.zeros(len(all_feat_cols), dtype=np.float64)

    for X, y in client_data.values():
        X_aligned = X.reindex(columns=all_feat_cols, fill_value=0)
        
        for col in X_aligned.columns:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors='coerce')
        X_aligned = X_aligned.fillna(0)
        
        total_n += len(X_aligned)
        sum_vec += X_aligned.to_numpy().sum(axis=0)
        sumsq_vec += (X_aligned.to_numpy()**2).sum(axis=0)

    global_mean = sum_vec / total_n
    global_var = (sumsq_vec / total_n) - global_mean**2
    global_std = np.sqrt(np.clip(global_var, 1e-8, None))

    np.save("/Users/faezehhosseini/phd_federated_learning_project/data/processed_data/global_mean.npy", global_mean)
    np.save("/Users/faezehhosseini/phd_federated_learning_project/data/processed_data/global_std.npy", global_std)
    pd.Series(all_feat_cols).to_csv("/Users/faezehhosseini/phd_federated_learning_project/data/processed_data/all_feat_cols.csv", index=False, header=False)
    
    print("Data preparation complete. Global scaler and feature list saved.")

# --- 3. The Flower Client (Hospital Logic) ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, algo: str, fedprox_mu: float, dp_config: dict):
        self.hospital_id = hospital_id
        self.algo = algo
        self.fedprox_mu = fedprox_mu
        self.dp_config = dp_config
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_scale_data()
        self.model = TorchLogisticRegression(n_features=self.X_train.shape[1])

    def load_and_scale_data(self):
        data_path = Path("/Users/faezehhosseini/phd_federated_learning_project/data/processed_data")
        X = pd.read_csv(data_path / f"X_{self.hospital_id}.csv")
        y = pd.read_csv(data_path / f"y_{self.hospital_id}.csv")
        
        all_feat_cols = pd.read_csv(data_path / "all_feat_cols.csv", header=None).iloc[:, 0].tolist()
        X = X.reindex(columns=all_feat_cols, fill_value=0)
        
        X = X.astype(float)
        
        global_mean = np.load(data_path / "global_mean.npy")
        global_std = np.load(data_path / "global_std.npy")
        X_scaled = (X - global_mean) / global_std
        
        X_np = X_scaled.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.float32).ravel()

        return train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_ds = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # --- THE FIX: Set model to train mode *before* attaching the Privacy Engine ---
        self.model.train()

        if self.dp_config.get("enable", False):
            print(f"[Client {self.hospital_id}] Training with Differential Privacy.")
            privacy_engine = PrivacyEngine()
            self.model, optimizer, train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=self.dp_config["noise_multiplier"],
                max_grad_norm=self.dp_config["max_grad_norm"],
            )

        global_params = None
        if self.algo == "fedprox":
            global_params = [p.detach().clone() for p in self.model.parameters()]

        for _ in range(5):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)

                if self.algo == "fedprox" and global_params:
                    prox_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        prox_term += (local_param - global_param).pow(2).sum()
                    loss += (self.fedprox_mu / 2) * prox_term

                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(self.X_test))
            probs = torch.sigmoid(logits).numpy()
            auc = roc_auc_score(self.y_test, probs)
        
        return 1.0 - auc, len(self.X_test), {"auc": auc}

# --- 4. Metrics Aggregation Function ---
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_auc_sum = sum([num_examples * m["auc"] for num_examples, m in metrics])
    return {"auc": weighted_auc_sum / total_examples}

# --- 5. Main script execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Learning Simulation")
    parser.add_argument("--mode", type=str, required=True, choices=["prepare", "server", "client"])
    parser.add_argument("--hospital_id", type=str, choices=["A", "B", "C"])
    parser.add_argument("--algo", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--fedprox_mu", type=float, default=0.1)
    parser.add_argument("--use_dp", action="store_true", help="Enable Differential Privacy")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="DP max gradient norm")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_federated_data()
    
    elif args.mode == "server":
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
        )
    
    elif args.mode == "client":
        if not args.hospital_id:
            raise ValueError("--hospital_id is required for client mode.")
        
        dp_config = {
            "enable": args.use_dp,
            "noise_multiplier": args.noise_multiplier,
            "max_grad_norm": args.max_grad_norm
        }
        
        client = FlowerClient(
            hospital_id=args.hospital_id,
            algo=args.algo,
            fedprox_mu=args.fedprox_mu,
            dp_config=dp_config
        )
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=client,
        )
