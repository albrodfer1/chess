"""Smoke tests for the network + MCTS + self-play pipeline."""

import chess

from chesszero.config import Config
from chesszero.mcts import Evaluator, run_mcts, select_move
from chesszero.network import ChessNet
from chesszero.selfplay import play_game


def _tiny_config() -> Config:
    return Config(
        num_res_blocks=1,
        num_filters=8,
        num_simulations=8,
        max_moves=6,
        temperature_moves=2,
        device="cpu",
    )


def test_mcts_returns_legal_move():
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    board = chess.Board()
    root = run_mcts(board, evaluator, config, add_noise=True)
    assert root.visit_count > 0

    move = select_move(root, temperature=0.0)
    assert move in board.legal_moves


def test_self_play_produces_labelled_examples():
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    examples = play_game(evaluator, config)
    assert len(examples) > 0
    for ex in examples:
        assert ex.state.shape == (19, 8, 8)
        assert ex.policy.shape == (config.action_size,)
        assert -1.0 <= ex.value <= 1.0
