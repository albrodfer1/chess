"""Saving and loading model checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .network import ChessNet


def save_checkpoint(path: str | Path, net: ChessNet, config: Config,
                    optimizer: torch.optim.Optimizer | None = None,
                    iteration: int = 0) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "config": asdict(config),
        "iteration": iteration,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | None = None) -> tuple[ChessNet, Config, dict]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    config = Config(**payload["config"])
    if device:
        config.device = device
    net = ChessNet(config).to(config.device)
    net.load_state_dict(payload["model_state"])
    return net, config, payload
