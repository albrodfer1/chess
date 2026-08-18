"""Residual policy-value network (AlphaZero style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """Two-headed network: a policy over the 4672 action space and a value.

    The value head outputs a scalar in [-1, 1] from the perspective of the
    player to move. Combined with MCTS, this is the value estimate referenced
    in requirement (3).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        f = config.num_filters

        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(f) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(f, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 64, config.action_size)

        # Value head
        self.value_conv = nn.Conv2d(f, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value
