"""Training step: fit the network to self-play search policies and outcomes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .network import ChessNet
from .replay_buffer import ReplayBuffer


def train_epochs(net: ChessNet, buffer: ReplayBuffer, optimizer: torch.optim.Optimizer,
                 config: Config) -> dict[str, float]:
    """Run several optimization epochs over samples from the replay buffer.

    Loss = MSE(value, outcome) + cross-entropy(policy_logits, search_policy).
    Weight decay (L2) is applied through the optimizer.
    """
    net.train()
    device = config.device

    total_policy_loss = 0.0
    total_value_loss = 0.0
    steps = 0

    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0}

    batches_per_epoch = max(1, len(buffer) // config.batch_size)

    for _ in range(config.epochs_per_iteration):
        for _ in range(batches_per_epoch):
            states, policies, values = buffer.sample(config.batch_size)

            states_t = torch.from_numpy(states).to(device)
            policies_t = torch.from_numpy(policies).to(device)
            values_t = torch.from_numpy(values).to(device)

            policy_logits, value_pred = net(states_t)

            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, values_t)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            steps += 1

    return {
        "policy_loss": total_policy_loss / steps,
        "value_loss": total_value_loss / steps,
    }
