# Chapter 17 — Scaling Up & Improvements

> **Course:** [Home](README.md) · **Prev:** [16. Debugging & Understanding Training](16-debugging-and-convergence.md) · **Next:** [18. Glossary & Further Reading](18-glossary-and-references.md)

**What you'll learn**
- Why real chess needs enormous compute, and what our modest defaults trade away
- Every scaling knob in `config.py` and the tradeoff each one makes
- Eight concrete engineering upgrades — each with the exact function and file to change
- A sense of which improvements give the most strength-per-hour

---

## 17.1 The compute reality of chess

DeepMind's AlphaZero trained on **thousands of TPUs**, playing tens of millions of self-play games. Chess has on the order of $10^{40}$+ reachable positions and a game tree far larger; the only way to cover enough of it is scale. Our project uses the *same algorithm*, but with defaults sized so the whole loop runs on a laptop in a reasonable time — which means it will play weakly out of the box. That's a deliberate tradeoff, and this chapter is about spending more compute (and cleverness) to buy strength.

Two truths to keep in mind:

1. **Self-play dominates the cost.** Each iteration runs roughly `games_per_iteration × plies_per_game × num_simulations` network forward passes. Training is comparatively cheap. So most speedups target self-play.
2. **Strength comes from the product of capacity, search, and data** — a bigger network with more simulations and more games. Maxing one while starving the others wastes the others.

All knobs below live in [`config.py`](../src/chesszero/config.py); all code pointers are to files in [`src/chesszero/`](../src/chesszero/).

## 17.2 The scaling knobs

Here is every dial in the `Config` dataclass, grouped by what it buys you.

### Capacity — how much the network *can* learn

```python
num_res_blocks: int = 6      # depth of the residual tower
num_filters: int = 128       # width (channels) of each conv layer
```

More blocks and filters give the network more capacity to represent chess knowledge — but every self-play move and every training step gets proportionally slower, and a bigger model needs more data to avoid overfitting. AlphaZero used ~19–40 residual blocks and 256 filters. Scaling `num_res_blocks` toward 10–20 and `num_filters` toward 256 is the path to a stronger evaluator, and is best done on a GPU. See [Chapter 10](10-the-neural-network.md) for what these control inside `ChessNet`.

### Search — how hard the agent *thinks* per move

```python
num_simulations: int = 100   # MCTS simulations per move
c_puct: float = 1.5          # exploration constant in PUCT
```

`num_simulations` is the single most direct strength dial: more simulations mean deeper, sharper search and a better move *and* a better policy target π for training ([Chapter 11](11-mcts-in-code.md)). AlphaZero used ~800. The cost is linear — doubling sims roughly doubles self-play time. `c_puct` balances exploiting the network's priors vs. exploring under-visited moves inside the tree; the default `1.5` is reasonable, but it interacts with how peaked your policy is.

### Data — how much experience feeds training

```python
games_per_iteration: int = 20
iterations: int = 10
replay_buffer_size: int = 50_000
```

More games per iteration and more iterations mean more, fresher experience. A larger `replay_buffer_size` remembers more history and fights forgetting ([Chapter 16](16-debugging-and-convergence.md)), at the cost of RAM and slightly staler data.

### Optimization — how the network digests that data

```python
batch_size: int = 256
learning_rate: float = 1e-3
weight_decay: float = 1e-4
epochs_per_iteration: int = 4
```

`batch_size` and `learning_rate` are the usual deep-learning dials ([Chapter 13](13-training-and-replay-buffer.md)); `weight_decay` is the L2 term that keeps weights small; `epochs_per_iteration` is how many passes over the buffer each training phase makes (more passes extract more from the data but risk overfitting to the current buffer).

### Device

```python
device: str = ""             # "" -> auto: cuda > mps > cpu
```

`default_device()` prefers `cuda`, then Apple's `mps`, then `cpu`. A GPU is the difference between minutes and hours per iteration. Override with the global `--device` flag if auto-detection is wrong.

### A rule of thumb for scaling up

Scale **search and data first** (they're cheap conceptually and give reliable gains), then **capacity** once you have the compute to feed a bigger network. A sensible "serious laptop/GPU" configuration might be `num_res_blocks=10`, `num_filters=256`, `num_simulations=400`, `games_per_iteration=100`, `iterations=50+`.

## 17.3 Engineering improvements

The defaults are also missing several optimizations and safeguards that real AlphaZero-style systems use. Each subsection names the concrete change and *where* it goes.

### (a) Tree reuse between moves

**Today:** `run_mcts` in [`mcts.py`](../src/chesszero/mcts.py) builds a brand-new tree from scratch on **every** move — it allocates a fresh `root`, calls `evaluate`, and expands from zero:

```python
def run_mcts(board, evaluator, config, add_noise=False):
    root = Node(prior=0.0)
    root.board = board.copy()
    priors, _ = evaluator.evaluate(root.board)
    _expand(root, priors)
    ...
```

But the subtree under the move you actually played is *already searched* — after you play move `m`, the child node `root.children[m]` is a fully-populated tree for the resulting position. Throwing it away wastes all those simulations.

**Improvement:** carry the tree across moves. After `select_move` picks `m`, keep `root.children[m]` as the new root (detach its `parent`, and re-add Dirichlet noise at the new root for self-play). This is a change to the *caller* — `play_game` in [`selfplay.py`](../src/chesszero/selfplay.py) and `MCTSAgent.choose_move` in [`agent.py`](../src/chesszero/agent.py) — plus a small helper in `mcts.py` that runs additional simulations on an existing root instead of always creating one. Payoff: often a large fraction of simulations are reused, effectively increasing search depth for free.

### (b) Batched / parallel leaf evaluation

**Today:** `Evaluator.evaluate` runs the network on **one** board at a time (`unsqueeze(0)` makes a batch of size 1). On a GPU that leaves most of the hardware idle.

**Improvement:** evaluate many leaves per network call. Two common approaches:
- **Parallel self-play games** on one process, batching the leaf boards that are waiting for evaluation across games into a single forward pass.
- **Virtual loss** within a single tree, letting several simulations descend to different leaves before a batched evaluation, then backing them up together.

This is the highest-leverage change for GPU throughput. It touches `Evaluator` (accept a list of boards, return a list of `(priors, value)`) and `run_mcts` (collect a batch of leaves before evaluating). It's more invasive, but self-play is where nearly all the time goes, so it pays off the most.

### (c) Perspective / mirrored board encoding

**Today:** `encode_board` in [`encoding.py`](../src/chesszero/encoding.py) uses **absolute** coordinates plus a side-to-move plane (plane 12). This was chosen for simplicity — it avoids having to mirror move indices ([Chapter 9](09-encoding-board-and-moves.md)) — but it forces the network to learn each pattern twice: once for White and once for the mirror-image Black version.

**Improvement:** encode from the **mover's perspective** — always orient the board so the side to move is "playing up," flipping ranks and swapping colours when it's Black's turn. Then White and Black positions that are strategically identical look identical to the network, roughly halving what it must learn. The catch, and the reason it wasn't done here, is that you must also mirror the *action* mapping (`move_to_index` / `index_to_move`) consistently so the policy still points at the right squares. Done carefully it's a clean win in sample efficiency.

### (d) Resignation and draw adjudication

**Today:** every self-play game in `play_game` runs until `board.is_game_over()` or the `max_moves` cap. Dead-lost and dead-drawn positions are played out pointlessly, burning simulations and (via the move cap) sometimes turning clearly-won games into truncated draws.

**Improvement:** use the search value to stop early. After `run_mcts`, `root.value` (from `mcts.py`) is the MCTS value for the side to move. Add to `play_game`:
- **Resignation:** if `root.value` stays below a threshold (e.g. `< -0.9`) for a few consecutive plies for the side to move, end the game as a loss for that side. This frees compute *and* sharpens the value signal by not diluting it with hopeless shuffling ([Chapter 16](16-debugging-and-convergence.md)).
- **Draw adjudication:** if the value hovers near 0 deep into a lifeless endgame, call it a draw rather than dragging to `max_moves`.

As AlphaZero did, keep a small fraction of games with resignation *disabled* to measure and bound false resignations. This is a localized change entirely inside `play_game`.

### (e) Arena gating (only promote a better network)

**Today:** `cmd_loop` trains the network in place and immediately uses it to generate the next iteration's self-play. If a training step happens to make the network *worse*, that regression pollutes the next batch of data.

**Improvement:** keep a "best" network and a "candidate." After training, use `cmd_eval`'s match logic (`_play_match` in [`cli.py`](../src/chesszero/cli.py)) to play the candidate against the current best for N games; **promote** the candidate to generate self-play only if it wins by a clear margin (AlphaGo Zero used ~55%). This is the classic guard against the forgetting/regression failure mode from [Chapter 16](16-debugging-and-convergence.md), and it reuses code you already have — you'd add a gating step between `train_epochs` and the next iteration's self-play, and track a separate `best.pt`.

### (f) Learning-rate schedule

**Today:** `_build` creates a plain `Adam` optimizer with a fixed `learning_rate`. AlphaZero used a **step schedule**, dropping the rate at milestones as training matured.

**Improvement:** wrap the optimizer in a `torch.optim.lr_scheduler` (e.g. `StepLR` or `CosineAnnealingLR`) and step it once per iteration in `cmd_loop`. A high rate early speeds initial learning; a lower rate later stabilizes it. This is a few lines in [`cli.py`](../src/chesszero/cli.py) around `_build` and the iteration loop.

### (g) Exploration tuning (Dirichlet & temperature)

**Today:** `dirichlet_alpha=0.3`, `dirichlet_epsilon=0.25`, and `temperature_moves=30` are fixed constants. AlphaZero tuned Dirichlet α to the game's branching factor (chess ≈ 0.3) and used a temperature *schedule* that drops to ~0 after the opening.

**Improvement:** the temperature step already exists in `play_game` (1.0 for the first `temperature_moves` plies, then 0.0). You can refine it into a smoother decay, and expose `dirichlet_alpha`/`dirichlet_epsilon` for experimentation. This is a low-risk way to fix the exploration failure modes in [Chapter 16](16-debugging-and-convergence.md). The relevant code is `_add_dirichlet_noise` in `mcts.py` and the temperature line in `play_game`.

### (h) Persisting the replay buffer and exporting PGN

**Today:** `ReplayBuffer` in [`replay_buffer.py`](../src/chesszero/replay_buffer.py) already has `save` and `load` methods — but `cmd_loop` never calls them, so the buffer is lost on exit and **not** restored by `--resume` ([Chapter 16](16-debugging-and-convergence.md)). Games are only saved as JSON for the viewer, and never as standard PGN.

**Improvements:**
- **Persist the buffer.** Call `buffer.save(path)` at the end of each iteration in `cmd_loop`, and `buffer.load(path)` when resuming. This makes `--resume` truly continuous instead of restarting training on an empty buffer. The methods already exist — you're just wiring them in.
- **Export PGN.** The viewer JSON captured in `play_game` (`_record_ply`) stores FENs and moves but not standard [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation). `python-chess` can build PGN via `chess.pgn`; adding an export lets you open self-play games in any chess GUI, run them through a reference engine, or share them.

### (i) Beyond a single machine

For serious training, the standard architecture separates **self-play workers** (many processes, each generating games with the latest network) from a single **trainer** (consuming their examples and publishing new checkpoints). Our `cmd_loop` runs both phases sequentially in one process; distributing them is the last big lever, and mostly a matter of a shared example queue and checkpoint store rather than new algorithm.

## 17.4 Which improvements pay off most?

Roughly, in decreasing strength-per-effort for a laptop/single-GPU setup:

| Improvement | Effort | Payoff | Why |
| --- | --- | --- | --- |
| Raise `num_simulations` + `iterations` | trivial (config) | high | direct strength; no code |
| (b) Batched leaf evaluation | high | very high | unlocks the GPU; self-play is the bottleneck |
| (a) Tree reuse | medium | high | free extra search depth every move |
| (c) Perspective encoding | medium | high | ~2× sample efficiency |
| (e) Arena gating | medium | medium–high | prevents regressions / forgetting |
| (d) Resignation | low | medium | cheaper, cleaner games |
| (h) Persist buffer / PGN | low | medium | continuity + inspectability |
| (f) LR schedule, (g) exploration tuning | low | small–medium | stability and diversity |

Start with the config knobs (§17.2) to confirm the pipeline strengthens at all, then invest in batched evaluation and tree reuse to make each hour of compute count.

---

## Key takeaways

- Chess demands scale; our defaults are laptop-sized on purpose. Strength comes from the **product** of capacity (`num_res_blocks`, `num_filters`), search (`num_simulations`), and data (`games_per_iteration`, `iterations`, `replay_buffer_size`).
- **Self-play is the bottleneck**, so the highest-leverage engineering wins are **batched leaf evaluation** (use the GPU) and **tree reuse** (stop throwing away searched subtrees).
- **Perspective encoding** roughly doubles sample efficiency; the only reason it isn't in the code is the care needed to mirror the action mapping.
- **Arena gating** and a larger replay buffer defend against regression and forgetting; **resignation/adjudication** make games cheaper and the value signal cleaner.
- Several upgrades are nearly free because the hooks already exist — `ReplayBuffer.save/load` just needs wiring into `cmd_loop`, and `_play_match` is ready to power arena gating.

## Exercises

1. Estimate the self-play cost of one iteration at the defaults (`games_per_iteration=20`, average game ~120 plies, `num_simulations=100`) in network forward passes. Now recompute with `num_simulations=400`. What does that do to your per-iteration wall-clock time?
2. Sketch the code change for **tree reuse**: what exactly would `play_game` keep after `select_move`, and what must you reset on the new root before searching again? (Hint: `parent`, and Dirichlet noise.)
3. Wire `ReplayBuffer.save`/`load` into `cmd_loop` so `--resume` restores the buffer. Where in the loop do you save, and where on resume do you load? What path convention would you use?
4. Explain why **perspective encoding** requires changing `move_to_index`/`index_to_move`, not just `encode_board`. What bug appears if you flip the board but forget to flip the moves?
5. Design an **arena gating** step using `_play_match`: how many games, what win-rate threshold to promote, and how would you track the "best" network separately from `latest.pt`?

---

> **Course:** [Home](README.md) · **Prev:** [16. Debugging & Understanding Training](16-debugging-and-convergence.md) · **Next:** [18. Glossary & Further Reading](18-glossary-and-references.md)
