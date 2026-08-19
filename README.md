# ChessZero

> Disclaimer: The code and documents of this project are AI-Generated

An AlphaZero-style reinforcement-learning chess agent. The model learns to play
purely from **self-play**, guided by **Monte Carlo Tree Search (MCTS)**, and it
is structurally incapable of selecting an **illegal move**.

It implements exactly the three properties requested:

1. **No illegal moves.** Move selection always goes through the legal-move set.
   MCTS only ever expands legal moves, and the network's policy is masked and
   renormalized over legal moves only (`encoding.legal_mask`, `mcts.Evaluator`).
2. **Learns by playing itself.** A single network plays both sides. Each game
   produces training examples labelled with the eventual game result
   (`selfplay.play_game`).
3. **Value comes from MCTS.** For every move, the network provides a prior
   policy and a leaf value; repeated PUCT simulations with value back-ups turn
   these into a refined search policy (visit counts) and search value
   (`mcts.run_mcts`).

## 📚 Learn how it works — the course

New to reinforcement learning? This repo doubles as a **"zero to hero" course**
that teaches RL from scratch by building this exact project. Start at
[`docs/README.md`](docs/README.md) and read the 18 lessons in order — from "what
is a reward?" through MDPs, MCTS, and AlphaZero, to a line-by-line tour of every
module here.

## Architecture

```
board (python-chess)
   │  encode_board  ─────────────► (19, 8, 8) tensor
   │
   ▼
ChessNet (ResNet tower)  ──►  policy logits (4672)  +  value (tanh, [-1,1])
   │
   ▼
Evaluator: mask illegal moves, softmax over legal moves only
   │
   ▼
MCTS (PUCT selection, value back-ups)  ──►  visit-count policy π  +  search value
   │
   ├─ self-play: sample move from π (temperature), store (state, π, result)
   ▼
Training: minimize  MSE(value, result) + cross-entropy(policy, π)
```

| Module | Responsibility |
| --- | --- |
| `encoding.py`  | board ↔ tensor, move ↔ index (4672 action space), **legal mask** |
| `network.py`   | residual policy + value network |
| `mcts.py`      | evaluator (legal masking) and PUCT Monte Carlo Tree Search |
| `selfplay.py`  | generate self-play games → training examples |
| `replay_buffer.py` | fixed-size example buffer |
| `train.py`     | one training pass (policy + value loss) |
| `agent.py`     | move-selection agent for play / evaluation |
| `cli.py`       | `loop`, `play`, `eval` commands |

## Training process

Training is a repeating **reinforcement-learning loop** driven by `cli.cmd_loop`.
Each *iteration* has two phases: self-play (the network is frozen and generates
games) and training (the network is updated on those games). It is **not** a
one-shot fill — the loop runs many iterations, each producing a stronger network
that in turn produces better self-play data.

```
_build(config)                         # 1. network initialized with RANDOM weights
for iteration in range(iterations):    # cli.cmd_loop
    # ── PHASE A: self-play (network FROZEN, net.eval) ─────────────
    for game in range(games_per_iteration):        # selfplay.play_game
        while not game_over and moves < max_moves:
            root = run_mcts(board, ...)             # 100 sims EVERY move
            π    = policy_from_visits(root)         # improved search policy
            store (encode_board(board), π, side_to_move)
            board.push(select_move(root, temperature))
        z = game_result(board)                      # +1 / 0 / -1, once at game end
        label every stored position with z (from its mover's perspective)
        buffer.add(examples)                        # replay_buffer
    # ── PHASE B: training (network UPDATED) ───────────────────────
    train_epochs(net, buffer, optimizer, config)    # train.py
        loss = MSE(value_pred, z) + cross_entropy(policy_logits, π)
    save_checkpoint(...)                            # checkpoints/model_iter_N.pt
```

Key points that are easy to get wrong:

- **MCTS runs on *every* move**, not at the end of the game. The end of the game
  only produces the outcome label `z`.
- **Weights update once per iteration** (Phase B), *after* a whole batch of games
  — never mid-game and never between games within a single iteration.
- **Two training signals.** The value head is trained on the real game outcome
  `z ∈ {+1, 0, −1}` (the only true *reward*). The policy head is trained to
  imitate the MCTS visit distribution `π` (a distillation of search, not a reward).
- **Bootstrapping.** During search, non-terminal leaves are scored by the value
  head, not by playing to the end — so low simulation counts still yield a value
  estimate. The real win/loss signal enters only through *completed* self-play
  games, which is why games must reach a decisive-or-drawn terminal state.

### Why self-play converges

The agent only ever faces itself, yet still improves, for three reasons:

1. **The reward is ground truth.** A checkmate is an objectively won game
   regardless of how weak both players are, so the outcome label is always correct.
2. **MCTS is a policy-improvement operator.** Search with lookahead is stronger
   than the raw network, so training toward `π` always pulls the network toward
   something better than its current self (classic policy iteration).
3. **Self-play is an automatic curriculum.** The opponent is always exactly the
   agent's own strength — never hopelessly hard nor trivially easy — which is the
   ideal difficulty for a learning signal.

Exploration is injected via **Dirichlet noise** at the MCTS root and **temperature**
sampling for the first `temperature_moves` plies, so games stay diverse and end
decisively often enough to keep the value signal alive.

## Setup

Requires Python 3.11–3.14 and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Usage

Run the reinforcement-learning loop (self-play → train → checkpoint, repeated).
Start small to sanity-check timing on a CPU:

```bash
poetry run chesszero loop --iterations 2 --games 4 --simulations 40
```

Checkpoints are written to `checkpoints/` (`model_iter_N.pt` and `latest.pt`).

### Saving games for the viewer

Add `--sample-games N` to record `N` self-play games, spread **evenly across the
whole run** (so you get early, middle, and late-training games to compare). Each
game is saved as JSON with per-move data — the played move, the network/MCTS
value, and the softmax evaluation of every legal move:

```bash
poetry run chesszero loop --iterations 8 --games 10 --sample-games 10
```

Games are written to `games/` by default (override with `--games-dir`). The
number is capped at what you request (e.g. 10), regardless of how many games are
actually played.

Play against a trained model:

```bash
poetry run chesszero play checkpoints/latest.pt --color white
```

Evaluate a model against another checkpoint or against a random player:

```bash
poetry run chesszero eval checkpoints/latest.pt --games 10                 # vs random
poetry run chesszero eval checkpoints/latest.pt --model-b checkpoints/model_iter_1.pt
```

Tune scale via `src/chesszero/config.py` (network size, simulations, games per
iteration, buffer size, learning rate, …).

## Game viewer (graphical interface)

A dependency-free browser viewer replays the saved games and shows, for every
move, the model's **softmax evaluation of each legal move** (network policy) side
by side with the **MCTS visit distribution**, plus the position value.

First produce some games with `--sample-games` (see above), then launch the
viewer:

```bash
poetry run chesszero viewer                 # serves games/ and opens a browser
poetry run chesszero viewer --games-dir games --port 8000
poetry run chesszero viewer --no-browser    # just serve; open the URL yourself
```

This starts a small local web server (default <http://127.0.0.1:8000/>) and opens
your browser. It needs no internet connection and no extra dependencies — the
board is rendered from FEN with Unicode pieces in plain JavaScript.

In the viewer you can:

- pick any saved game from the dropdown (labelled by iteration and result);
- step through moves with the ⏮ ◀ ▶ ⏭ buttons, the slider, or the ← / → arrow
  keys, or hit **Play** (spacebar) to auto-advance;
- see the **played move highlighted** on the board (from/to squares);
- read the per-move **evaluation bars**: blue = policy softmax probability the
  network assigned to each move, green = fraction of MCTS visits it received. The
  move actually played is marked ✓;
- watch the **value gauge** (side-to-move perspective, network and MCTS values).

Because the games are sampled across training, flipping between an early game and
a late one is a quick visual check that the agent is actually learning — the
evaluation mass should concentrate on stronger moves over time.

## Tests

The suite lives in `tests/` and covers the two things most likely to break
silently: the encoding layer and the search/self-play pipeline.

Run everything:

```bash
poetry run pytest
```

Useful variants:

```bash
poetry run pytest -v                        # verbose: list every test
poetry run pytest tests/test_encoding.py    # one file
poetry run pytest -k roundtrip              # tests whose name matches a pattern
poetry run pytest -x -q                     # stop at first failure, quiet output
```

What the tests check:

| File | What it verifies |
| --- | --- |
| `tests/test_encoding.py` | action space is 4672; board encodes to `(19, 8, 8)`; **every legal move round-trips** through `move_to_index`/`index_to_move`; the legal mask matches `board.legal_moves`; move indices are unique per position |
| `tests/test_mcts.py` | MCTS always returns a **legal** move; self-play produces correctly shaped, correctly labelled training examples (uses a tiny CPU network so it runs in seconds) |

The MCTS/self-play tests intentionally use a tiny network (`num_filters=8`,
`num_simulations=8`) and force `device="cpu"`, so the whole suite finishes in a
few seconds without a GPU.

## Notes on scale

Chess is enormous, so a strong agent needs a lot of self-play, larger networks,
and many MCTS simulations — typically far more compute than a single CPU. The
defaults here are deliberately modest so the full loop runs end-to-end on a
laptop; increase `num_res_blocks`, `num_filters`, `num_simulations`, and
`games_per_iteration` (ideally on a GPU) to actually grow playing strength.
