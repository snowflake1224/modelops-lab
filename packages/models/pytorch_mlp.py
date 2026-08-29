"""PyTorch MLP for tabular classification."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from packages.models.base import BaseModel, ModelConfig


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.BatchNorm1d(dim), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PyTorchMLPModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._network: MLP | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "PyTorchMLPModel":
        start = time.perf_counter()
        hp = self.config.hyperparameters
        dropout = hp.get("dropout", 0.2)
        epochs = hp.get("epochs", 30)
        batch_size = hp.get("batch_size", 256)
        lr = hp.get("learning_rate", 1e-3)

        torch.manual_seed(self.config.random_seed)
        hidden_dims = hp.get("hidden_dims", [128, 64, 32])
        if not isinstance(hidden_dims, list):
            hidden_dims = [128, 64, 32]
        network = MLP(int(x.shape[1]), hidden_dims, float(dropout)).to(self._device)
        self._network = network

        # Class-weighted BCE loss
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(network.parameters(), lr=float(lr))

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=int(batch_size),
            shuffle=True,
        )

        network.train()
        for _ in range(int(epochs)):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                optimizer.zero_grad()
                logits = network(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.model = network
        self.training_duration_s = time.perf_counter() - start
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("PyTorch model is not fitted")
        network = self._network
        network.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).to(self._device)
            logits = network(tensor)
            proba = torch.sigmoid(logits).cpu().numpy()
        return np.asarray(proba)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self._network.state_dict() if self._network else None,
                "input_dim": self._network.network[0].in_features if self._network else 0,
                "feature_names": self.feature_names_,
                "training_duration_s": self.training_duration_s,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PyTorchMLPModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["config"])
        hp = data["config"].hyperparameters
        hidden = hp.get("hidden_dims", [128, 64, 32])
        dropout = hp.get("dropout", 0.2)
        instance._network = MLP(data["input_dim"], hidden, dropout)
        if data["state_dict"]:
            instance._network.load_state_dict(data["state_dict"])
        instance.model = instance._network
        instance.feature_names_ = data.get("feature_names", [])
        instance.training_duration_s = data.get("training_duration_s", 0.0)
        return instance
