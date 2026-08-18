# Chapter 8 — Project Setup & Architecture

> **Course:** [Home](README.md) · **Prev:** [7. The AlphaZero Algorithm](07-the-alphazero-algorithm.md) · **Next:** [9. Encoding the Board & Moves](09-encoding-board-and-moves.md)

**What you'll learn**
- How the project is packaged with **Poetry**, and why that matters
- What every line of `pyproject.toml` does — dependencies, entry point, build backend
- The `src/` layout and the full directory tree
- A module map: what each file in `src/chesszero/` is responsible for
- The `Config` dataclass — every hyperparameter in one place — and automatic device selection
- The four CLI subcommands you'll use for the rest of the course

---

Welcome to **Part III**. Parts I and II gave you the ideas: rewards and returns,
value functions, self-play, Monte Carlo Tree Search, and the AlphaZero loop.
From here on we build the real thing, one module at a time, always tying the
code back to the theory.

This chapter is the "lay of the land." We won't write any chess logic yet — we'll
set up the project, install it, and learn where everything lives so that the next
seven chapters have a home to slot into.

## 8.1 Prerequisites

You need two things installed:

- **Python 3.11–3.14.** The project targets modern Python (it uses `X | None`
  type syntax and `from __future__ import annotations`).
- **[Poetry](https://python-poetry.org/) 2.x**, a dependency and packaging tool
  for Python.

That's it. Everything else — PyTorch, python-chess, NumPy — Poetry installs for
you.

## 8.2 Why Poetry?

You may be used to `pip install` and a `requirements.txt`. Poetry does the same
job but adds structure that matters for a real project:

| What Poetry gives you | Why it helps |
| --- | --- |
| A single `pyproject.toml` | one file describes the package, its dependencies, and its tooling |
| A **lockfile** (`poetry.lock`) | pins *exact* versions so every machine installs the identical environment |
| An isolated **virtual environment** | the project's packages don't pollute your system Python |
| **Scripts** / entry points | `poetry run chesszero ...` becomes a real command |
| Dependency **groups** | keep dev-only tools (like `pytest`) out of production installs |

The practical upshot: one command, `poetry install`, gives you a reproducible
environment, and `poetry run <cmd>` runs anything inside it.

## 8.3 `pyproject.toml`, line by line

Here is the whole file:

```toml
[project]
name = "chesszero"
version = "0.1.0"
description = "AlphaZero-style reinforcement learning agent that learns chess from self-play using Monte Carlo Tree Search."
authors = [{ name = "Alberto", email = "alberto.rfernandez4@gmail.com" }]
readme = "README.md"
requires-python = ">=3.11,<3.15"
dependencies = [
    "torch (>=2.0)",
    "python-chess (>=1.11)",
    "numpy (>=1.24)",
]

[project.scripts]
chesszero = "chesszero.cli:main"

[tool.poetry]
packages = [{ include = "chesszero", from = "src" }]

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"

[build-system]
requires = ["poetry-core>=2.0"]
build-backend = "poetry.core.masonry.api"
```

Let's unpack each block.

### The `[project]` table

This is the standard, tool-agnostic metadata (defined by [PEP 621](https://peps.python.org/pep-0621/)).

- `requires-python = ">=3.11,<3.15"` — the supported interpreter range.
- `dependencies` — the three libraries the project *cannot run without*:

| Dependency | Role in the project |
| --- | --- |
| **`torch`** (PyTorch) | the neural network — layers, autograd, the training optimizer, GPU/MPS acceleration |
| **`python-chess`** | the chess *rules engine* — legal move generation, board state, game-over detection. We never re-implement chess; we lean on this. |
| **`numpy`** | fast array math for the board/move encodings that feed the network |

⚠️ Note the design decision hiding in that table: **we do not implement chess
ourselves.** `python-chess` is the environment (the "rules of the game" from
[Chapter 2](02-rl-fundamentals.md)). This is why the agent can *never* attempt an
illegal move for a reason unrelated to the network — the legal move list always
comes from a correct, battle-tested library. More on that in
[Chapter 9](09-encoding-board-and-moves.md).

### The `[project.scripts]` entry point

```toml
[project.scripts]
chesszero = "chesszero.cli:main"
```

This one line is what makes `poetry run chesszero ...` work. It says: "create a
command called `chesszero` that calls the `main` function in the `chesszero.cli`
module." Every command in this course — `loop`, `play`, `eval`, `viewer` — flows
through that `main` function (see [`cli.py`](../src/chesszero/cli.py)).

### The `[tool.poetry]` packages and the `src/` layout

```toml
[tool.poetry]
packages = [{ include = "chesszero", from = "src" }]
```

This tells Poetry the importable package `chesszero` lives under the `src/`
directory. This is the so-called **`src/` layout**, and it's a deliberate choice:

- It prevents accidentally importing the package from the working directory
  instead of the installed version — so your tests exercise the *installed*
  package, catching packaging mistakes early.
- It keeps the repository root tidy: source in `src/`, tests in `tests/`, docs in
  `docs/`.

### The dev dependency group

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
```

`pytest` is only needed for development (running the test suite), not for someone
who just wants to run a trained model. Putting it in a `dev` group keeps it out
of production installs.

### The build system

```toml
[build-system]
requires = ["poetry-core>=2.0"]
build-backend = "poetry.core.masonry.api"
```

Standard boilerplate telling Python *how* to build the package (using
`poetry-core`). You rarely touch this.

## 8.4 Installing and running

From the project root:

```bash
poetry install          # create the virtualenv and install everything
poetry run pytest       # run the test suite (should be all green)
poetry run chesszero --help
```

`poetry run chesszero --help` prints the available subcommands. You should see
`loop`, `play`, `eval`, and `viewer`. If that works, your environment is ready.

## 8.5 The directory tree

Here is the whole project laid out:

```
chess/
├── pyproject.toml            # packaging + dependencies (this chapter)
├── README.md                 # quick-start for the whole project
├── docs/                     # this course
│   ├── README.md
│   ├── 01-introduction.md
│   └── ...                   # chapters 02–18
├── src/
│   └── chesszero/            # the importable Python package
│       ├── __init__.py
│       ├── config.py         # all hyperparameters (this chapter)
│       ├── encoding.py       # board/move ↔ tensors, legal mask   → ch 9
│       ├── network.py        # ChessNet: the neural network        → ch 10
│       ├── mcts.py           # Evaluator + Monte Carlo Tree Search → ch 11
│       ├── selfplay.py       # self-play game generation           → ch 12
│       ├── replay_buffer.py  # training-example buffer             → ch 13
│       ├── train.py          # the training step                   → ch 13
│       ├── checkpoint.py     # save/load models                    → ch 14
│       ├── agent.py          # move-selection for real play        → ch 15
│       ├── cli.py            # the command-line interface          → ch 14/15
│       ├── viewer.py         # the browser game-viewer server      → ch 15
│       └── viewer/
│           └── index.html    # the viewer front-end (vanilla JS)   → ch 15
└── tests/
    ├── test_encoding.py
    ├── test_mcts.py
    └── test_recording.py
```

## 8.6 The module map

Every file in `src/chesszero/` has one clear job. Keep this table handy — it's
the index for the rest of Part III.

| Module | Responsibility | Chapter |
| --- | --- | --- |
| [`config.py`](../src/chesszero/config.py) | one dataclass holding every hyperparameter, plus device selection | this chapter |
| [`encoding.py`](../src/chesszero/encoding.py) | board → `(19,8,8)` tensor; move ↔ index in the 4672-action space; the **legal mask** | [9](09-encoding-board-and-moves.md) |
| [`network.py`](../src/chesszero/network.py) | `ChessNet` — a residual tower with a **policy head** and a **value head** | [10](10-the-neural-network.md) |
| [`mcts.py`](../src/chesszero/mcts.py) | `Evaluator` (network inference + legal masking) and `run_mcts` (PUCT search) | [11](11-mcts-in-code.md) |
| [`selfplay.py`](../src/chesszero/selfplay.py) | `play_game` — generate one self-play game and its training examples | [12](12-self-play-in-code.md) |
| [`replay_buffer.py`](../src/chesszero/replay_buffer.py) | a fixed-size buffer of `(state, policy, value)` examples | [13](13-training-and-replay-buffer.md) |
| [`train.py`](../src/chesszero/train.py) | `train_epochs` — one training pass (policy + value loss) | [13](13-training-and-replay-buffer.md) |
| [`checkpoint.py`](../src/chesszero/checkpoint.py) | save/load network + optimizer + config to a `.pt` file | [14](14-the-reinforcement-loop.md) |
| [`agent.py`](../src/chesszero/agent.py) | `MCTSAgent` — wraps the net + MCTS to pick moves for real play | [15](15-playing-evaluating-viewer.md) |
| [`cli.py`](../src/chesszero/cli.py) | argument parsing and the `loop` / `play` / `eval` / `viewer` commands | [14](14-the-reinforcement-loop.md)/[15](15-playing-evaluating-viewer.md) |
| [`viewer.py`](../src/chesszero/viewer.py) | a tiny local web server that serves saved games to the browser viewer | [15](15-playing-evaluating-viewer.md) |

Notice how the table mirrors the system diagram from [§1.5](01-introduction.md):
`selfplay.py` and `train.py` are the two phases, `mcts.py` and `network.py` are
the brains, and `encoding.py` is the translator between chess and math.

## 8.7 `config.py` — one place for every knob

RL systems have a lot of dials. Rather than scatter magic numbers across the
code, the project collects them all in a single [`Config`](../src/chesszero/config.py)
dataclass:

```python
@dataclass
class Config:
    # --- Encoding (fixed by the representation) ---
    input_planes: int = INPUT_PLANES        # 19
    action_size: int = ACTION_SIZE          # 4672

    # --- Network ---
    num_res_blocks: int = 6
    num_filters: int = 128

    # --- MCTS ---
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Self-play ---
    max_moves: int = 200
    temperature_moves: int = 30   # sample proportional to visits for first N plies
    games_per_iteration: int = 20

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 4
    replay_buffer_size: int = 50_000

    # --- Loop / IO ---
    iterations: int = 10
    checkpoint_dir: str = "checkpoints"
    device: str = ""
```

The comments group the knobs by the subsystem that uses them, and each group
maps to a chapter:

- **Encoding** (`input_planes`, `action_size`) — fixed by the representation in
  [Chapter 9](09-encoding-board-and-moves.md). You don't tune these.
- **Network** (`num_res_blocks`, `num_filters`) — the size of the neural network
  ([Chapter 10](10-the-neural-network.md)). Bigger = smarter but slower.
- **MCTS** (`num_simulations`, `c_puct`, `dirichlet_*`) — how the search behaves
  ([Chapter 11](11-mcts-in-code.md)). `num_simulations` is how *hard* it thinks
  per move.
- **Self-play** (`max_moves`, `temperature_moves`, `games_per_iteration`) — how
  games are generated ([Chapter 12](12-self-play-in-code.md)).
- **Training** (`batch_size`, `learning_rate`, `weight_decay`,
  `epochs_per_iteration`, `replay_buffer_size`) — the optimizer's behaviour
  ([Chapter 13](13-training-and-replay-buffer.md)).
- **Loop / IO** (`iterations`, `checkpoint_dir`, `device`) — the outer loop and
  where things get saved ([Chapter 14](14-the-reinforcement-loop.md)).

### Automatic device selection

The last piece of `config.py` picks the fastest available hardware:

```python
def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

And the dataclass fills it in automatically if you didn't specify one:

```python
    def __post_init__(self) -> None:
        if not self.device:
            self.device = default_device()
```

So on an NVIDIA machine you get `cuda`, on a modern Mac you get `mps` (Apple's
Metal GPU backend), and everywhere else you fall back to `cpu`. You can always
override it with the global `--device` flag on the CLI.

📐 *Why is `action_size` in `Config` if it's fixed?* Because the network's policy
head needs to know its output width at construction time, and threading it
through the config keeps `network.py` from importing encoding constants directly.
It's a small decoupling convenience.

## 8.8 The CLI, from a distance

Everything you do with the project goes through [`cli.py`](../src/chesszero/cli.py).
Its `main()` builds an argument parser with four subcommands:

```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chesszero", description=__doc__)
    parser.add_argument("--device", default="", help="cpu / cuda / mps (auto by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_loop = sub.add_parser("loop", help="self-play + train reinforcement loop")
    ...
    p_play = sub.add_parser("play", help="play against a trained model")
    ...
    p_eval = sub.add_parser("eval", help="evaluate a model vs another model or random")
    ...
    p_view = sub.add_parser("viewer", help="open the browser game viewer")
    ...
```

Here's what each does — we'll build them properly in later chapters:

| Command | What it does | Chapter |
| --- | --- | --- |
| `chesszero loop` | run the self-play → train → checkpoint reinforcement loop | [14](14-the-reinforcement-loop.md) |
| `chesszero play` | play a game against a trained model from your terminal | [15](15-playing-evaluating-viewer.md) |
| `chesszero eval` | pit a model against another model (or a random player) | [15](15-playing-evaluating-viewer.md) |
| `chesszero viewer` | launch a browser viewer to replay saved self-play games | [15](15-playing-evaluating-viewer.md) |

The pattern `argparse` uses here is worth noting: each subparser calls
`set_defaults(func=...)` to attach the function that handles it, and `main`
finishes with:

```python
    args = parser.parse_args(argv)
    args.func(args)
```

So parsing the arguments *also* selects which command function to run. Clean and
extensible — adding a new command is just adding a new subparser.

## 8.9 Where we go next

You now know the shape of the project: how it's packaged, where each file lives,
and what every hyperparameter controls. The next chapter dives into the very
first thing any RL system needs — a way to turn the environment's **state** (a
chess position) and **actions** (moves) into numbers a neural network can
consume, while guaranteeing the agent can never choose an illegal move.

---

## Key takeaways

- The project is a proper Python package managed by **Poetry**; `poetry install`
  gives you a reproducible environment and `poetry run chesszero ...` runs the CLI.
- `pyproject.toml` declares three runtime dependencies — **torch** (network),
  **python-chess** (rules engine), **numpy** (encodings) — and an entry point that
  maps the `chesszero` command to `cli:main`.
- The code uses a **`src/` layout**; every module has one responsibility,
  summarized in the module-map table.
- All hyperparameters live in one `Config` dataclass, grouped by subsystem, with
  automatic `cuda → mps → cpu` device selection.
- The CLI exposes four subcommands — `loop`, `play`, `eval`, `viewer` — each built
  in a later chapter.

## Exercises

1. Run `poetry install` and then `poetry run chesszero --help`. List the four
   subcommands and, from the help text alone, guess what each does.
2. Open `config.py`. Which single hyperparameter would you change to make MCTS
   "think harder" on every move? Which would make the neural network *larger*?
3. Why does the project depend on `python-chess` instead of implementing chess
   rules itself? Connect your answer to the "no illegal moves" requirement from
   [Chapter 1](01-introduction.md).
4. The `src/` layout puts `chesszero` under `src/`. What problem does this prevent
   compared to putting the package directly in the project root? (Hint: think
   about what gets imported when you run tests from the root directory.)
5. Trace the path from typing `poetry run chesszero loop` to a Python function
   being called. Which two lines in `main()` are responsible for choosing and
   invoking the right command?

---

> **Course:** [Home](README.md) · **Prev:** [7. The AlphaZero Algorithm](07-the-alphazero-algorithm.md) · **Next:** [9. Encoding the Board & Moves](09-encoding-board-and-moves.md)
