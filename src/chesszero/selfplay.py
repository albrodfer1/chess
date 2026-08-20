"""Self-play game generation (requirement 2: the model learns from itself)."""

from __future__ import annotations

import chess

from .config import Config
from .encoding import encode_board
from .mcts import Evaluator, is_terminal, policy_from_visits, run_mcts, select_move
from .replay_buffer import Example


def play_game(evaluator: Evaluator, config: Config, record: bool = False):
    """Play one full self-play game and return training examples.

    The same network plays both sides. At each position we run MCTS, record the
    resulting search policy, then play a move (sampled with temperature early on
    for exploration, greedy later). When the game ends, every stored position is
    labelled with the final result relative to its mover.

    Always returns ``(examples, info)``. ``info`` is a JSON-serializable summary
    of the finished game: ``result`` (+1/0/-1, white's perspective), ``winner``
    ("white wins"/"black wins"/"draw"), ``result_str`` ("1-0"/"0-1"/"1/2-1/2"),
    ``num_plies`` and ``termination`` (the reason the game ended: "checkmate",
    "threefold repetition", "fivefold repetition", "max moves", ...).

    If ``record`` is True, ``info`` additionally carries a ``moves`` list with
    per-ply data (FEN, the move played, the network value, and the softmax
    evaluation of every legal move) suitable for the browser viewer.
    """
    board = chess.Board()
    history: list[tuple] = []  # (state, policy, side_to_move)
    ply_records: list[dict] = []

    move_number = 0
    while not is_terminal(board) and move_number < config.max_moves:
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

    info = {
        "result": result,                 # +1 / 0 / -1 (white perspective)
        "winner": _winner_str(result),
        "result_str": board.result(claim_draw=True),
        "num_plies": move_number,
        "termination": _termination_reason(board, move_number, config),
    }

    if not record:
        return examples, info

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

    info["moves"] = ply_records
    return examples, info


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


def _winner_str(result: float) -> str:
    if result > 0:
        return "white wins"
    if result < 0:
        return "black wins"
    return "draw"


def _termination_reason(board: chess.Board, move_number: int, config: Config) -> str:
    """Why the game ended, based on the final board (not a claim look-ahead).

    If the position isn't terminal under our rules, the loop can only have
    stopped by hitting the move cap. Otherwise we report the specific rule,
    checking fivefold before threefold since fivefold implies threefold.
    """
    if not is_terminal(board):
        return "max moves"
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.is_repetition(3):
        return "threefold repetition"
    if board.is_seventyfive_moves():
        return "seventy-five-move rule"
    return "unknown"


def _game_result(board: chess.Board) -> float:
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0
