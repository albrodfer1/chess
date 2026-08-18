"""Move-selection agent used for evaluation and human play."""

from __future__ import annotations

import chess

from .config import Config
from .mcts import Evaluator, run_mcts, select_move
from .network import ChessNet


class MCTSAgent:
    """Chooses moves by running MCTS and picking the most-visited root child.

    Because MCTS only ever expands legal moves, this agent can never return an
    illegal move (requirement 1).
    """

    def __init__(self, net: ChessNet, config: Config, simulations: int | None = None) -> None:
        self.config = config
        if simulations is not None:
            self.config = Config(**{**config.__dict__})
            self.config.num_simulations = simulations
        self.evaluator = Evaluator(net, self.config.device)

    def choose_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        net = self.evaluator.net
        was_training = net.training
        net.eval()
        try:
            root = run_mcts(board, self.evaluator, self.config, add_noise=False)
        finally:
            net.train(was_training)
        return select_move(root, temperature=temperature)
