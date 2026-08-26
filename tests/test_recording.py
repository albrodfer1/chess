"""Tests for game recording, sparse sampling, and the viewer manifest."""

import json

import chess

from chesszero.agent import MCTSAgent
from chesszero.checkpoint import save_checkpoint
from chesszero.cli import _evenly_spaced_indices, _play_match, _save_game, main
from chesszero.config import Config
from chesszero.mcts import Evaluator
from chesszero.network import ChessNet
from chesszero.selfplay import play_game
from chesszero.viewer import _build_manifest


def _tiny_config() -> Config:
    return Config(num_res_blocks=1, num_filters=8, num_simulations=8,
                  max_moves=6, temperature_moves=2, device="cpu")


def test_evenly_spaced_indices():
    assert _evenly_spaced_indices(100, 0) == set()
    assert _evenly_spaced_indices(4, 10) == {0, 1, 2, 3}   # count >= total
    idx = _evenly_spaced_indices(100, 5)
    assert idx == {0, 25, 50, 74, 99}
    assert len(_evenly_spaced_indices(50, 10)) == 10


def test_play_game_record_shape():
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    examples, record = play_game(evaluator, config, record=True)
    assert len(examples) > 0

    assert set(record) >= {"result", "result_str", "num_plies", "termination", "moves"}
    # A trailing terminal position is always appended for the final board.
    assert record["moves"][-1]["terminal"] is True

    first = record["moves"][0]
    assert chess.Board().is_valid()
    assert first["turn"] == "white"
    assert first["played"] is not None
    assert -1.0 <= first["value"] <= 1.0
    for ev in first["evaluations"]:
        assert {"uci", "san", "policy", "visits"} <= set(ev)
        assert 0.0 <= ev["policy"] <= 1.0
    # Evaluations are sorted by network policy, descending.
    policies = [e["policy"] for e in first["evaluations"]]
    assert policies == sorted(policies, reverse=True)


def test_save_game_and_manifest(tmp_path):
    config = _tiny_config()
    net = ChessNet(config)
    net.eval()
    evaluator = Evaluator(net, "cpu")

    _, record = play_game(evaluator, config, record=True)
    path = _save_game(tmp_path, global_index=7, iteration=2, record=record)

    saved = json.loads(path.read_text())
    assert saved["game_index"] == 7
    assert saved["iteration"] == 2

    manifest = _build_manifest(tmp_path)
    assert len(manifest) == 1
    assert manifest[0]["game_index"] == 7
    assert manifest[0]["file"] == path.name


def test_eval_match_record_is_viewer_compatible():
    config = _tiny_config()
    config.num_simulations = 1
    config.max_moves = 2
    agent_a = MCTSAgent(ChessNet(config), config)
    agent_b = MCTSAgent(ChessNet(config), config)

    result, record = _play_match(agent_a, agent_b, a_is_white=True,
                                 max_moves=config.max_moves, record=True)

    assert result in {-1.0, 0.0, 1.0}
    assert record is not None
    assert record["num_plies"] == 2
    assert len(record["moves"]) == 3  # two moves plus final board state
    assert record["moves"][-1]["terminal"] is True
    assert all(move["played"] is not None for move in record["moves"][:-1])


def test_eval_cli_saves_recorded_game(tmp_path):
    config = _tiny_config()
    config.num_simulations = 1
    config.max_moves = 2
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, ChessNet(config), config)
    games_dir = tmp_path / "eval-games"

    main([
        "--device", "cpu", "eval", str(checkpoint), "--model-b", str(checkpoint),
        "--games", "1", "--simulations", "1", "--record-games-dir", str(games_dir),
    ])

    saved = list(games_dir.glob("*.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text())
    assert record["moves"][-1]["terminal"] is True
