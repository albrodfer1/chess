"""Self-play game generation (requirement 2: the model learns from itself)."""

from __future__ import annotations

import chess

from .config import Config
from .encoding import encode_board
from .mcts import Evaluator, policy_from_visits, run_mcts, select_move
from .replay_buffer import Example


def play_game(evaluator: Evaluator, config: Config, record: bool = False):
    """Play one full self-play game and return training examples.

    The same network plays both sides. At each position we run MCTS, record the
    resulting search policy, then play a move (sampled with temperature early on
    for exploration, greedy later). When the game ends, every stored position is
    labelled with the final result relative to its mover.

    If ``record`` is True, returns ``(examples, game_record)`` where
    ``game_record`` is a JSON-serializable dict with per-ply data (FEN, the move
    played, the network value, and the softmax evaluation of every legal move)
    suitable for the browser viewer. Otherwise returns just ``examples``.
    """
    board = chess.Board()
    history: list[tuple] = []  # (state, policy, side_to_move)
    ply_records: list[dict] = []

    move_number = 0
    while not board.is_game_over() and move_number < config.max_moves:
        root = run_mcts(board, evaluator, config, add_noise=True)
        if not root.children:
            break

        temperature = 1.0 if move_number < config.temperature_moves else 0.0
        policy = policy_from_visits(root, temperature=1.0)
        history.append((encode_board(board), policy, board.turn))

        move = select_move(root, temperature=temperature)

        if record:
            ply_records.append(_record_ply(board, root, move, evaluator, move_number))

        board.push(move)
        move_number += 1

    result = _game_result(board)  # +1 white win, -1 black win, 0 draw/unfinished

    examples: list[Example] = []
    for state, policy, side_to_move in history:
        value = result if side_to_move == chess.WHITE else -result
        examples.append(Example(state=state, policy=policy, value=float(value)))

    if not record:
        return examples

    # Append the final position so the viewer can show the finished board.
    ply_records.append({
        "ply": move_number,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "played": None,
        "value": None,
        "mcts_value": None,
        "evaluations": [],
        "terminal": True,
    })

    game_record = {
        "result": result,                 # +1 / 0 / -1 (white perspective)
        "result_str": board.result(claim_draw=True),
        "num_plies": move_number,
        "termination": _termination_str(board),
        "moves": ply_records,
    }
    return examples, game_record


def _record_ply(board: chess.Board, root, move: chess.Move,
                evaluator: Evaluator, ply: int) -> dict:
    """Capture per-ply data for the viewer.

    ``evaluations`` is the raw network policy softmax over legal moves (the
    "evaluation of each move"), paired with the MCTS visit fraction. We re-run a
    clean evaluation so the reported softmax is free of the root Dirichlet noise
    that MCTS injects during self-play.
    """
    priors, net_value = evaluator.evaluate(board)
    total_visits = sum(c.visit_count for c in root.children.values()) or 1

    evaluations = []
    for mv, child in root.children.items():
        evaluations.append({
            "uci": mv.uci(),
            "san": board.san(mv),
            "policy": round(priors.get(mv, 0.0), 5),          # network softmax
            "visits": round(child.visit_count / total_visits, 5),  # MCTS policy
        })
    evaluations.sort(key=lambda e: e["policy"], reverse=True)

    return {
        "ply": ply,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "played": {"uci": move.uci(), "san": board.san(move)},
        "value": round(net_value, 5),        # network value, side-to-move view
        "mcts_value": round(root.value, 5),  # MCTS search value, side-to-move view
        "evaluations": evaluations[:12],     # top 12 by network probability
        "terminal": False,
    }


def _termination_str(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "move-limit"
    return outcome.termination.name.lower()


def _game_result(board: chess.Board) -> float:
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0
