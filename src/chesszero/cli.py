"""Command-line interface: train via self-play, play, and evaluate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import chess
import torch

from .agent import MCTSAgent
from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .mcts import Evaluator
from .network import ChessNet
from .replay_buffer import ReplayBuffer
from .selfplay import play_games_batch
from .train import train_epochs
from .viewer import run_viewer


def _build(config: Config) -> tuple[ChessNet, torch.optim.Optimizer]:
    net = ChessNet(config).to(config.device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return net, optimizer


def _evenly_spaced_indices(total: int, count: int) -> set[int]:
    """Pick ``count`` game indices spread evenly across ``total`` games.

    Used to sample games sparsely throughout the whole run (early, mid, late
    training) rather than clustering them.
    """
    if count <= 0 or total <= 0:
        return set()
    if count >= total:
        return set(range(total))
    return {round(i * (total - 1) / (count - 1)) for i in range(count)}


def _git(*args: str) -> str:
    """Run a git command from the repo and return its stripped stdout."""
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "git is required to train (runs are namespaced by commit hash), "
            f"but `git {' '.join(args)}` failed: {exc}"
        ) from exc


def _require_clean_worktree() -> str:
    """Return the current short commit hash, refusing to run if the tree is dirty.

    Training is namespaced by commit so a run is reproducible from its hash;
    uncommitted changes would make that hash a lie. Gitignored artifacts (the
    checkpoint/game output dirs) don't show up here, so they don't block a run.
    """
    dirty = _git("status", "--porcelain")
    if dirty:
        raise SystemExit(
            "Refusing to train with uncommitted changes — commit or stash them "
            "so the run is reproducible from its git hash.\n" + dirty
        )
    return _git("rev-parse", "--short", "HEAD")


def _save_game(games_dir: Path, global_index: int, iteration: int,
               record: dict) -> Path:
    games_dir.mkdir(parents=True, exist_ok=True)
    record = {"game_index": global_index, "iteration": iteration, **record}
    path = games_dir / f"game_{global_index:04d}_iter{iteration:02d}.json"
    path.write_text(json.dumps(record))
    return path


def cmd_loop(args: argparse.Namespace) -> None:
    """Self-play + train loop: the full reinforcement-learning cycle."""
    config = Config(device=args.device or "")
    if args.simulations:
        config.num_simulations = args.simulations
    if args.games:
        config.games_per_iteration = args.games
    if args.iterations:
        config.iterations = args.iterations
    if args.parallel_games:
        config.selfplay_batch_size = args.parallel_games

    # Reproducibility: refuse to train on a dirty tree, and namespace every
    # artifact of this run by the commit it was trained at.
    git_hash = _require_clean_worktree()

    print(f"Device: {config.device} | sims/move: {config.num_simulations} "
          f"| games/iter: {config.games_per_iteration} "
          f"| parallel: {config.selfplay_batch_size} | git: {git_hash}")

    net, optimizer = _build(config)
    buffer = ReplayBuffer(config.replay_buffer_size)
    start_iter = 0

    if args.resume and Path(args.resume).exists():
        net, config, payload = load_checkpoint(args.resume, device=config.device)
        net, optimizer = net, torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_iter = payload.get("iteration", 0)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    # Per-commit output folders + a config.json recording exactly what we ran.
    # (--checkpoint-dir wins even on --resume, so training can be redirected to
    # e.g. a mounted Google Drive folder.)
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    ckpt_dir = Path(config.checkpoint_dir) / git_hash
    games_dir = Path(args.games_dir) / git_hash
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_dir / "config.json"
    config_path.write_text(
        json.dumps({"git_hash": git_hash, **asdict(config)}, indent=2)
    )
    print(f"Run dir: {ckpt_dir}/ (config -> {config_path})")

    # Decide which games to record, spread evenly across the whole run.
    total_games = config.iterations * config.games_per_iteration
    sample_set = _evenly_spaced_indices(total_games, args.sample_games)
    if sample_set:
        print(f"Sampling {len(sample_set)} game(s) sparsely across {total_games} "
              f"to '{games_dir}/' for the viewer")
    global_index = 0

    for iteration in range(start_iter, start_iter + config.iterations):
        t0 = time.time()

        net.eval()
        evaluator = Evaluator(net, config.device)
        new_examples = 0
        outcomes = {"white wins": 0, "black wins": 0, "draw": 0}
        reasons: dict[str, int] = {}

        base_index = global_index
        record_indices = {g for g in range(config.games_per_iteration)
                          if base_index + g in sample_set}

        def on_game_done(g: int, examples, info) -> None:
            nonlocal new_examples
            global_idx = base_index + g
            outcomes[info["winner"]] += 1
            reasons[info["termination"]] = reasons.get(info["termination"], 0) + 1

            line = (f"  [iter {iteration}] self-play game {g + 1}/"
                    f"{config.games_per_iteration}: {len(examples)} positions "
                    f"| {info['winner']} ({info['termination']}) "
                    f"in {info['num_plies']} plies")
            if global_idx in sample_set:
                path = _save_game(games_dir, global_idx, iteration + 1, info)
                line += f" (saved -> {path})"
            print(line, flush=True)

            buffer.add(examples)
            new_examples += len(examples)

        play_games_batch(evaluator, config,
                         num_games=config.games_per_iteration,
                         batch_size=config.selfplay_batch_size,
                         record_indices=record_indices,
                         on_game_done=on_game_done)
        global_index = base_index + config.games_per_iteration

        reason_str = ", ".join(f"{n} {r}" for r, n in sorted(reasons.items()))
        print(f"  [iter {iteration}] results: "
              f"W {outcomes['white wins']} / B {outcomes['black wins']} / "
              f"D {outcomes['draw']}  ({reason_str})", flush=True)

        stats = train_epochs(net, buffer, optimizer, config)

        ckpt_path = ckpt_dir / f"model_iter_{iteration + 1}.pt"
        save_checkpoint(ckpt_path, net, config, optimizer, iteration=iteration + 1)
        save_checkpoint(ckpt_dir / "latest.pt", net, config, optimizer, iteration=iteration + 1)

        dt = time.time() - t0
        print(f"[iter {iteration}] positions+{new_examples} buffer={len(buffer)} "
              f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
              f"time={dt:.1f}s -> {ckpt_path}", flush=True)


def cmd_play(args: argparse.Namespace) -> None:
    """Play a game against a trained model from the terminal."""
    net, config, _ = load_checkpoint(args.checkpoint, device=args.device or None)
    if args.simulations:
        config.num_simulations = args.simulations
    net.eval()
    agent = MCTSAgent(net, config)

    board = chess.Board()
    human_is_white = args.color == "white"
    print("You play", args.color, "- enter moves in UCI (e.g. e2e4) or SAN (e.g. Nf3).")
    print("Type 'quit' to exit.\n")

    while not board.is_game_over():
        print(board, "\n")
        if board.turn == (chess.WHITE if human_is_white else chess.BLACK):
            move = _read_human_move(board)
            if move is None:
                return
        else:
            print("Thinking...", flush=True)
            move = agent.choose_move(board, temperature=0.0)
            print(f"Model plays: {board.san(move)}\n")
        board.push(move)

    print(board, "\n")
    print("Result:", board.result(), "-", board.outcome())


def _read_human_move(board: chess.Board) -> chess.Move | None:
    while True:
        raw = input("Your move: ").strip()
        if raw.lower() in ("quit", "exit"):
            return None
        move = None
        try:
            move = board.parse_san(raw)
        except ValueError:
            try:
                move = chess.Move.from_uci(raw)
            except ValueError:
                move = None
        if move is not None and move in board.legal_moves:
            return move
        print("Illegal or unparseable move; try again.")


def cmd_eval(args: argparse.Namespace) -> None:
    """Play games between two checkpoints (or a checkpoint vs. random)."""
    net_a, config_a, _ = load_checkpoint(args.model_a, device=args.device or None)
    net_a.eval()
    agent_a = MCTSAgent(net_a, config_a, simulations=args.simulations or None)

    agent_b = None
    if args.model_b:
        net_b, config_b, _ = load_checkpoint(args.model_b, device=args.device or None)
        net_b.eval()
        agent_b = MCTSAgent(net_b, config_b, simulations=args.simulations or None)

    wins_a = wins_b = draws = 0
    for i in range(args.games):
        a_is_white = i % 2 == 0
        result = _play_match(agent_a, agent_b, a_is_white, config_a.max_moves)
        if result == 0:
            draws += 1
        elif (result == 1) == a_is_white:
            wins_a += 1
        else:
            wins_b += 1
        print(f"game {i + 1}/{args.games}: A={wins_a} B={wins_b} draws={draws}", flush=True)

    label_b = "model_b" if args.model_b else "random"
    print(f"\nmodel_a wins: {wins_a} | {label_b} wins: {wins_b} | draws: {draws}")


def _play_match(agent_a: MCTSAgent, agent_b, a_is_white: bool, max_moves: int) -> float:
    import random

    board = chess.Board()
    moves = 0
    while not board.is_game_over() and moves < max_moves:
        a_turn = board.turn == (chess.WHITE if a_is_white else chess.BLACK)
        if a_turn:
            move = agent_a.choose_move(board, temperature=0.0)
        elif agent_b is not None:
            move = agent_b.choose_move(board, temperature=0.0)
        else:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        moves += 1

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0


def cmd_viewer(args: argparse.Namespace) -> None:
    """Launch the browser-based game viewer."""
    run_viewer(games_dir=args.games_dir, port=args.port,
               open_browser=not args.no_browser)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chesszero", description=__doc__)
    parser.add_argument("--device", default="", help="cpu / cuda / mps (auto by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_loop = sub.add_parser("loop", help="self-play + train reinforcement loop")
    p_loop.add_argument("--iterations", type=int, default=0)
    p_loop.add_argument("--games", type=int, default=0, help="self-play games per iteration")
    p_loop.add_argument("--simulations", type=int, default=0, help="MCTS sims per move")
    p_loop.add_argument("--parallel-games", type=int, default=0,
                        help="self-play games to run concurrently with batched "
                             "NN evaluation (bigger = better GPU utilization)")
    p_loop.add_argument("--resume", default="", help="checkpoint path to resume from")
    p_loop.add_argument("--sample-games", type=int, default=0,
                        help="save this many games, spread evenly across the run "
                             "(e.g. 10), for the viewer")
    p_loop.add_argument("--games-dir", default="games",
                        help="directory to write sampled game JSON files")
    p_loop.add_argument("--checkpoint-dir", default="",
                        help="directory for checkpoints (e.g. a mounted Google "
                             "Drive folder); overrides the config default")
    p_loop.set_defaults(func=cmd_loop)

    p_play = sub.add_parser("play", help="play against a trained model")
    p_play.add_argument("checkpoint")
    p_play.add_argument("--color", choices=["white", "black"], default="white")
    p_play.add_argument("--simulations", type=int, default=0)
    p_play.set_defaults(func=cmd_play)

    p_eval = sub.add_parser("eval", help="evaluate a model vs another model or random")
    p_eval.add_argument("model_a")
    p_eval.add_argument("--model-b", default="")
    p_eval.add_argument("--games", type=int, default=10)
    p_eval.add_argument("--simulations", type=int, default=0)
    p_eval.set_defaults(func=cmd_eval)

    p_view = sub.add_parser("viewer", help="open the browser game viewer")
    p_view.add_argument("--games-dir", default="games", help="directory of saved games")
    p_view.add_argument("--port", type=int, default=8000)
    p_view.add_argument("--no-browser", action="store_true",
                        help="don't auto-open a browser window")
    p_view.set_defaults(func=cmd_viewer)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
