"""Central hyperparameter configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .encoding import ACTION_SIZE, INPUT_PLANES


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # --- Encoding (fixed by the representation) ---
    input_planes: int = INPUT_PLANES
    action_size: int = ACTION_SIZE

    # --- Network ---
    num_res_blocks: int = 6
    num_filters: int = 128

    # --- MCTS ---
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Self-play ---
    max_moves: int = 200
    temperature_moves: int = 30   # sample proportional to visits for first N plies
    games_per_iteration: int = 20

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 4
    replay_buffer_size: int = 50_000

    # --- Loop / IO ---
    iterations: int = 10
    checkpoint_dir: str = "checkpoints"
    device: str = ""

    def __post_init__(self) -> None:
        if not self.device:
            self.device = default_device()
