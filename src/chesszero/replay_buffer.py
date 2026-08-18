"""Fixed-size replay buffer of self-play training examples."""

from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Example:
    state: np.ndarray   # (19, 8, 8) float32
    policy: np.ndarray  # (4672,) float32 search policy
    value: float        # game outcome from this state's mover perspective


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Example] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, examples: list[Example]) -> None:
        self.buffer.extend(examples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([e.state for e in batch])
        policies = np.stack([e.policy for e in batch])
        values = np.array([e.value for e in batch], dtype=np.float32)
        return states, policies, values

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.buffer.extend(pickle.load(f))
