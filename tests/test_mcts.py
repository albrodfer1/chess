"""Smoke tests for the network + MCTS + self-play pipeline."""

import chess

from chesszero.config import Config
from chesszero.mcts import Evaluator, is_terminal, run_mcts, run_mcts_batch, select_move
from chesszero.network import ChessNet
from chesszero.selfplay import play_game, play_games_batch


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


def test_threefold_repetition_is_terminal():
    board = chess.Board()
    # Shuffle both knights out and back: each full cycle repeats the position.
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for move in cycle * 2:
        assert not is_terminal(board)  # not yet a threefold
        board.push_uci(move)
    # The starting position has now occurred three times.
    assert board.is_repetition(3)
    assert is_terminal(board)
    # A plain is_game_over() would miss it (threefold is only claimable).
    assert not board.is_game_over()


def test_batched_search_matches_single_search():
    """Batched MCTS must be identical to per-board MCTS (no noise = deterministic)."""
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    board = chess.Board()
    single = run_mcts(board, evaluator, config, add_noise=False)
    batched = run_mcts_batch([board.copy(), board.copy()], evaluator, config,
                             add_noise=False)

    def visits(root):
        return sorted((m.uci(), c.visit_count) for m, c in root.children.items())

    # Independent trees, same input -> identical visit distributions.
    assert visits(batched[0]) == visits(single)
    assert visits(batched[1]) == visits(single)


def test_play_games_batch():
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    done = []
    results = play_games_batch(evaluator, config, num_games=5, batch_size=2,
                               record_indices={0},
                               on_game_done=lambda i, ex, info: done.append(i))

    assert len(results) == 5
    assert all(r is not None for r in results)          # refill pool played them all
    assert sorted(done) == [0, 1, 2, 3, 4]              # callback fired once per game
    assert "moves" in results[0][1]                     # index 0 was recorded
    assert "moves" not in results[1][1]                 # others were not
    for examples, info in results:
        assert info["winner"] in {"white wins", "black wins", "draw"}
        for ex in examples:
            assert ex.state.shape == (config.input_planes, 8, 8)


def test_self_play_produces_labelled_examples():
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    examples, info = play_game(evaluator, config)
    assert len(examples) > 0
    assert info["winner"] in {"white wins", "black wins", "draw"}
    assert info["termination"]  # a non-empty reason string
    for ex in examples:
        assert ex.state.shape == (config.input_planes, 8, 8)
        assert ex.policy.shape == (config.action_size,)
        assert -1.0 <= ex.value <= 1.0
