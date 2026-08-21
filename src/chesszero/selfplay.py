"""Self-play game generation (requirement 2: the model learns from itself)."""

from __future__ import annotations

import chess

from .config import Config
from .encoding import encode_board
from .mcts import (
    Evaluator,
    is_terminal,
    policy_from_visits,
    run_mcts,
    run_mcts_batch,
    select_move,
)
from .replay_buffer import Example


class _SelfPlayGame:
    """Mutable state for one in-progress self-play game.

    Holds everything a game accumulates as it is stepped ply by ply, so that
    many games can be driven together (batched self-play) without tangling
    their state.
    """

    def __init__(self, record: bool = False) -> None:
        self.board = chess.Board()
        self.history: list[tuple] = []   # (state, policy, side_to_move)
        self.ply_records: list[dict] = []
        self.move_number = 0
        self.record = record
        self.done = False


def _advance_game(game: _SelfPlayGame, root, evaluator: Evaluator,
                  config: Config) -> None:
    """Play one ply of ``game`` from its already-searched ``root``.

    Records the search policy, samples/greedy-picks a move (temperature early,
    greedy later), pushes it, and marks the game done on a terminal position or
    the move cap.
    """
    if not root.children:
        game.done = True
        return

    temperature = 1.0 if game.move_number < config.temperature_moves else 0.0
    policy = policy_from_visits(root, temperature=1.0)
    game.history.append((encode_board(game.board), policy, game.board.turn))

    move = select_move(root, temperature=temperature)
    if game.record:
        game.ply_records.append(
            _record_ply(game.board, root, move, evaluator, game.move_number)
        )

    game.board.push(move)
    game.move_number += 1
    if is_terminal(game.board) or game.move_number >= config.max_moves:
        game.done = True


def _finalize_game(game: _SelfPlayGame, config: Config):
    """Turn a finished game into ``(examples, info)`` (see ``play_game``)."""
    board = game.board
    result = _game_result(board)  # +1 white win, -1 black win, 0 draw/unfinished

    examples: list[Example] = []
    for state, policy, side_to_move in game.history:
        value = result if side_to_move == chess.WHITE else -result
        examples.append(Example(state=state, policy=policy, value=float(value)))

    info = {
        "result": result,                 # +1 / 0 / -1 (white perspective)
        "winner": _winner_str(result),
        "result_str": board.result(claim_draw=True),
        "num_plies": game.move_number,
        "termination": _termination_reason(board, game.move_number, config),
    }

    if not game.record:
        return examples, info

    # Append the final position so the viewer can show the finished board.
    game.ply_records.append({
        "ply": game.move_number,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "played": None,
        "value": None,
        "mcts_value": None,
        "evaluations": [],
        "terminal": True,
    })

    info["moves"] = game.ply_records
    return examples, info


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
    game = _SelfPlayGame(record=record)
    while not game.done:
        root = run_mcts(game.board, evaluator, config, add_noise=True)
        _advance_game(game, root, evaluator, config)
    return _finalize_game(game, config)


def play_games_batch(evaluator: Evaluator, config: Config, num_games: int,
                     batch_size: int | None = None,
                     record_indices: set[int] | None = None,
                     on_game_done=None):
    """Generate ``num_games`` self-play games with batched network evaluation.

    Up to ``batch_size`` games are kept in flight at once and stepped in
    lockstep, so all their MCTS evaluations batch into single forward passes.
    A finished game is immediately replaced by a fresh one until ``num_games``
    have been played — this "refill pool" keeps the batch (and the GPU) full
    even though chess games vary a lot in length.

    Returns a list of ``(examples, info)`` in game-start order. If
    ``on_game_done`` is given it is called as ``on_game_done(index, examples,
    info)`` the moment each game finishes (finish order, not start order), which
    the training loop uses for live logging.
    """
    record_indices = record_indices or set()
    batch_size = batch_size or num_games
    batch_size = max(1, min(batch_size, num_games))

    results: list = [None] * num_games
    active: list[tuple[int, _SelfPlayGame]] = []
    next_start = 0

    def refill() -> None:
        nonlocal next_start
        while next_start < num_games and len(active) < batch_size:
            active.append(
                (next_start, _SelfPlayGame(record=next_start in record_indices))
            )
            next_start += 1

    refill()
    while active:
        roots = run_mcts_batch([g.board for _, g in active], evaluator, config,
                               add_noise=True)
        for (index, game), root in zip(active, roots):
            _advance_game(game, root, evaluator, config)
            if game.done:
                results[index] = _finalize_game(game, config)
                if on_game_done is not None:
                    on_game_done(index, *results[index])

        if any(game.done for _, game in active):
            active = [(i, g) for i, g in active if not g.done]
            refill()

    return results


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
