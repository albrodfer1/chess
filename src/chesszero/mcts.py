"""Monte Carlo Tree Search guided by the policy-value network.

MCTS is where the agent's *value* estimates come from (requirement 3): the
network gives a prior policy and a leaf value, and repeated simulations with
PUCT selection + value backups refine both into a search policy (visit counts)
and a search value. Only legal moves are ever expanded (requirement 1).
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch

from .config import Config
from .encoding import ACTION_SIZE, encode_board, move_to_index
from .network import ChessNet


class Evaluator:
    """Wraps the network to score a single board with a legal-masked policy."""

    def __init__(self, net: ChessNet, device: str) -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """Return ({legal_move: prior_prob}, value_for_side_to_move)."""
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        logits = logits[0].cpu().numpy()

        moves = list(board.legal_moves)
        if not moves:
            return {}, float(value.item())

        indices = np.fromiter((move_to_index(m) for m in moves), dtype=np.int64)
        move_logits = logits[indices]
        # Softmax over legal moves only -> illegal moves get zero probability.
        move_logits -= move_logits.max()
        priors = np.exp(move_logits)
        priors /= priors.sum()

        return {m: float(p) for m, p in zip(moves, priors)}, float(value.item())


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "parent", "move", "board")

    def __init__(self, prior: float, parent: "Node | None" = None,
                 move: chess.Move | None = None) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.parent = parent
        self.move = move
        self.board: chess.Board | None = None

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def terminal_value(board: chess.Board) -> float:
    """Game result from the perspective of the side to move at `board`."""
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    # If it's a win, the side to move can't be the winner (they were mated).
    return 1.0 if outcome.winner == board.turn else -1.0


def _expand(node: Node, priors: dict[chess.Move, float]) -> None:
    for move, prob in priors.items():
        node.children[move] = Node(prior=prob, parent=node, move=move)


def _select_child(node: Node, c_puct: float) -> Node:
    best_score = -float("inf")
    best_child = None
    sqrt_total = math.sqrt(node.visit_count)
    for child in node.children.values():
        # Child value is from the child's mover perspective (the opponent),
        # so the parent scores it as -child.value.
        q = -child.value
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def _add_dirichlet_noise(root: Node, config: Config) -> None:
    moves = list(root.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(moves))
    eps = config.dirichlet_epsilon
    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = (1 - eps) * child.prior + eps * float(n)


def run_mcts(board: chess.Board, evaluator: Evaluator, config: Config,
             add_noise: bool = False) -> Node:
    """Run simulations from `board` and return the (expanded) root node."""
    root = Node(prior=0.0)
    root.board = board.copy()

    priors, _ = evaluator.evaluate(root.board)
    _expand(root, priors)
    if add_noise and root.children:
        _add_dirichlet_noise(root, config)

    for _ in range(config.num_simulations):
        node = root
        search_path = [root]

        # Selection: descend until we reach an unexpanded node.
        while node.expanded:
            node = _select_child(node, config.c_puct)
            search_path.append(node)

        # Lazily materialize the board for the reached leaf.
        if node.board is None:
            node.board = node.parent.board.copy()
            node.board.push(node.move)

        if node.board.is_game_over():
            value = terminal_value(node.board)
        else:
            priors, value = evaluator.evaluate(node.board)
            _expand(node, priors)

        # Backup: alternate sign each ply back up the tree.
        for path_node in reversed(search_path):
            path_node.visit_count += 1
            path_node.value_sum += value
            value = -value

    return root


def policy_from_visits(root: Node, temperature: float = 1.0) -> np.ndarray:
    """Build a (4672,) search-policy vector from child visit counts."""
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if visits.sum() == 0:
        return policy

    if temperature <= 1e-6:
        # Deterministic: all weight on the most-visited move.
        best = moves[int(visits.argmax())]
        policy[move_to_index(best)] = 1.0
        return policy

    scaled = visits ** (1.0 / temperature)
    scaled /= scaled.sum()
    for move, p in zip(moves, scaled):
        policy[move_to_index(move)] = p
    return policy


def select_move(root: Node, temperature: float = 0.0) -> chess.Move:
    """Pick a move from the root: greedy (temp 0) or sampled by visit policy."""
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if temperature <= 1e-6:
        return moves[int(visits.argmax())]

    probs = visits ** (1.0 / temperature)
    probs /= probs.sum()
    return moves[int(np.random.choice(len(moves), p=probs))]
