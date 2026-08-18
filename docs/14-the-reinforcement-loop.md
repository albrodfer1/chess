# Chapter 14 — The Reinforcement Loop & Checkpoints

> **Course:** [Home](README.md) · **Prev:** [13. Training & the Replay Buffer](13-training-and-replay-buffer.md) · **Next:** [15. Playing, Evaluating & the Viewer](15-playing-evaluating-viewer.md)

**What you'll learn**
- How self-play (Chapter 12) and training (Chapter 13) are stitched into one repeating loop
- The exact structure of `cmd_loop` — the code realization of *generalized policy iteration* from [Chapter 7](07-the-alphazero-algorithm.md)
- Why the network is *frozen* during self-play and *updated* only once per iteration
- How games get sampled sparsely across the run for the viewer
- What a `.pt` checkpoint actually contains, how resuming works, and one security caveat

---

## 14.1 The whole loop in one place

Everything you have learned so far — encoding ([Ch 9](09-encoding-board-and-moves.md)),
the network ([Ch 10](10-the-neural-network.md)), MCTS ([Ch 11](11-mcts-in-code.md)),
self-play ([Ch 12](12-self-play-in-code.md)), and training ([Ch 13](13-training-and-replay-buffer.md))
— comes together in a single function: `cmd_loop` in [`cli.py`](../src/chesszero/cli.py).

Recall the picture from [Chapter 7](07-the-alphazero-algorithm.md): AlphaZero is
**generalized policy iteration**. We alternate between *using* the current network
to generate good data (self-play with MCTS) and *improving* the network from that
data (supervised training on the search results). `cmd_loop` is literally that
alternation, wrapped in a `for` loop over **iterations**:

```
                     ┌─────────────────────────────────────────────┐
                     │        one ITERATION of cmd_loop             │
                     └─────────────────────────────────────────────┘

   PHASE A — SELF-PLAY (network FROZEN, net.eval())
   ┌──────────────────────────────────────────────────────────┐
   │ for g in range(games_per_iteration):                      │
   │     examples = play_game(evaluator, config)   ← Ch 12     │
   │     buffer.add(examples)                       ← Ch 13    │
   └──────────────────────────────────────────────────────────┘
                              │  replay buffer full of (state, π, z)
                              ▼
   PHASE B — TRAINING (network UPDATED)
   ┌──────────────────────────────────────────────────────────┐
   │ stats = train_epochs(net, buffer, optimizer, config)      │
   │        minimize MSE(value, z) + CE(policy, π)  ← Ch 13    │
   └──────────────────────────────────────────────────────────┘
                              │  a stronger network
                              ▼
   SAVE — save_checkpoint(model_iter_N.pt) and latest.pt
                              │
                              └──► next iteration uses the improved network
```

The key mental model: **an iteration is a batch of games followed by a batch of
training.** Do that ten times (the default) and each round of games is played by
a better network than the last.

## 14.2 Building the network and optimizer

Before the loop runs, we need a network and something to train it with. That's
`_build`:

```python
def _build(config: Config) -> tuple[ChessNet, torch.optim.Optimizer]:
    net = ChessNet(config).to(config.device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return net, optimizer
```

Two things happen here:

1. **`ChessNet(config).to(config.device)`** — instantiate the residual network
   from [Chapter 10](10-the-neural-network.md) and move it onto the chosen device
   (`cuda`, `mps`, or `cpu`; see [Chapter 8](08-project-setup-and-architecture.md)).
2. **`torch.optim.Adam(...)`** — the optimizer that will apply gradient updates.
   Note `weight_decay=config.weight_decay` (default `1e-4`). That is the L2
   regularization term — the `c‖θ‖²` from the AlphaZero loss in
   [Chapter 7](07-the-alphazero-algorithm.md). It is applied *by the optimizer*,
   not written into the loss in [`train.py`](../src/chesszero/train.py).

📐 **Why Adam?** Adam adapts a per-parameter learning rate from running estimates
of the gradient's mean and variance. It is forgiving of hyperparameter choices,
which is convenient here where we are not doing an elaborate learning-rate
schedule. [Chapter 17](17-scaling-and-improvements.md) discusses when a schedule
(e.g. SGD with momentum + step decay, as in the original AlphaZero) is worth it.

## 14.3 Setting up the loop

Here is the opening of `cmd_loop`, which turns command-line flags into a `Config`
and prepares the loop's state:

```python
def cmd_loop(args: argparse.Namespace) -> None:
    """Self-play + train loop: the full reinforcement-learning cycle."""
    config = Config(device=args.device or "")
    if args.simulations:
        config.num_simulations = args.simulations
    if args.games:
        config.games_per_iteration = args.games
    if args.iterations:
        config.iterations = args.iterations

    print(f"Device: {config.device} | sims/move: {config.num_simulations} "
          f"| games/iter: {config.games_per_iteration}")

    ckpt_dir = Path(config.checkpoint_dir)
    net, optimizer = _build(config)
    buffer = ReplayBuffer(config.replay_buffer_size)
    start_iter = 0
```

The pattern `if args.simulations:` means "only override the default if the user
passed a nonzero value on the command line." So `--simulations 400` bumps MCTS
strength, while omitting it keeps the `Config` default of `100`. Everything the
loop needs is now in hand:

- `config` — all hyperparameters ([Chapter 8](08-project-setup-and-architecture.md)).
- `net`, `optimizer` — the thing we're training.
- `buffer` — the [replay buffer](13-training-and-replay-buffer.md) that holds
  training examples.

## 14.4 Resuming from a checkpoint

Training a real chess agent takes far longer than one sitting. The `--resume`
flag lets you pick up where a previous run stopped:

```python
    if args.resume and Path(args.resume).exists():
        net, config, payload = load_checkpoint(args.resume, device=config.device)
        net, optimizer = net, torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_iter = payload.get("iteration", 0)
        print(f"Resumed from {args.resume} at iteration {start_iter}")
```

We reload the network weights *and* the optimizer state (Adam's running moment
estimates — restoring them avoids a jarring "cold" first step), and we set
`start_iter` so checkpoint filenames keep counting up instead of overwriting.

> ⚠️ **The replay buffer is not restored.** `buffer` was just created empty and
> stays empty on resume. The first post-resume iteration therefore trains only on
> freshly generated games. For short runs this is fine; for serious training you
> would persist the buffer too (the `ReplayBuffer.save`/`load` methods exist for
> exactly this — see [Chapter 13](13-training-and-replay-buffer.md) and the
> improvement discussion in [Chapter 16](16-debugging-and-convergence.md)).

## 14.5 Choosing which games to save for the viewer

Before iterating, `cmd_loop` decides which self-play games to record for the
[game viewer](15-playing-evaluating-viewer.md). We do not want to save *every*
game (thousands of JSON files); we want a **handful spread evenly across the whole
run**, so we can compare early, middle, and late play and literally watch the
agent improve.

```python
    # Decide which games to record, spread evenly across the whole run.
    total_games = config.iterations * config.games_per_iteration
    sample_set = _evenly_spaced_indices(total_games, args.sample_games)
    games_dir = Path(args.games_dir)
    if sample_set:
        print(f"Sampling {len(sample_set)} game(s) sparsely across {total_games} "
              f"to '{games_dir}/' for the viewer")
    global_index = 0
```

The sampling logic is a small, self-contained helper:

```python
def _evenly_spaced_indices(total: int, count: int) -> set[int]:
    if count <= 0 or total <= 0:
        return set()
    if count >= total:
        return set(range(total))
    return {round(i * (total - 1) / (count - 1)) for i in range(count)}
```

📐 **How the spacing works.** For `count` samples over `total` games, we place
markers at fractions `i / (count - 1)` of the way through, for `i = 0 … count-1`,
and scale by `total - 1`. So `--sample-games 10` over 200 games picks game
indices `0, 22, 44, …, 199` — the first, the last, and eight evenly in between.
Edge cases are handled: asking for zero (or when nothing will be saved) returns an
empty set; asking for more samples than games saves them all.

`global_index` is a counter that increases across *every* game of *every*
iteration, so a game's identity is stable regardless of which iteration it falls
in — that is what we test membership of `sample_set` against.

## 14.6 The iteration loop: Phase A (self-play)

Now the heart of it. For each iteration we first generate games:

```python
    for iteration in range(start_iter, start_iter + config.iterations):
        t0 = time.time()

        net.eval()
        evaluator = Evaluator(net, config.device)
        new_examples = 0
        for g in range(config.games_per_iteration):
            if global_index in sample_set:
                examples, record = play_game(evaluator, config, record=True)
                path = _save_game(games_dir, global_index, iteration + 1, record)
                print(f"  [iter {iteration}] self-play game {g + 1}/"
                      f"{config.games_per_iteration}: {len(examples)} positions "
                      f"(saved -> {path})", flush=True)
            else:
                examples = play_game(evaluator, config)
                print(f"  [iter {iteration}] self-play game {g + 1}/"
                      f"{config.games_per_iteration}: {len(examples)} positions", flush=True)
            buffer.add(examples)
            new_examples += len(examples)
            global_index += 1
```

Read this carefully — it embodies a point that trips up almost everyone:

- **`net.eval()`** puts the network in evaluation mode (batch-norm uses running
  statistics, no dropout). Crucially, *no gradients are computed and no weights
  change here.* The network is **frozen** for the entire self-play phase.
- We build one `Evaluator` (the legal-masking wrapper from
  [Chapter 11](11-mcts-in-code.md)) around the frozen net and reuse it for every
  game and every MCTS simulation in this iteration.
- If this game's `global_index` is in the sample set, we call
  `play_game(..., record=True)`, which additionally returns a rich per-ply
  `record` (see [Chapter 12](12-self-play-in-code.md)), and we write it to disk
  with `_save_game`. Otherwise we call the plain `play_game`.
- Either way, the resulting `examples` (the `(state, π, z)` tuples) go into the
  replay `buffer`.

> ⚠️ **Common misconception: "the network learns after each game" or "each move."**
> It does not. Throughout Phase A the network is fixed. MCTS runs on *every* move
> (that is where the search happens), but the *weights* do not move until Phase B.
> Learning happens **once per iteration**, after a whole batch of games. This is
> exactly the self-play → train separation from
> [Chapter 7](07-the-alphazero-algorithm.md).

## 14.7 The iteration loop: Phase B (training) and saving

Once the games are in the buffer, we improve the network and checkpoint it:

```python
        stats = train_epochs(net, buffer, optimizer, config)

        ckpt_path = ckpt_dir / f"model_iter_{iteration + 1}.pt"
        save_checkpoint(ckpt_path, net, config, optimizer, iteration=iteration + 1)
        save_checkpoint(ckpt_dir / "latest.pt", net, config, optimizer, iteration=iteration + 1)

        dt = time.time() - t0
        print(f"[iter {iteration}] positions+{new_examples} buffer={len(buffer)} "
              f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
              f"time={dt:.1f}s -> {ckpt_path}", flush=True)
```

- **`train_epochs(...)`** ([Chapter 13](13-training-and-replay-buffer.md)) samples
  minibatches from the buffer and runs several epochs of gradient descent. This is
  the *only* place the weights change. It returns mean losses for logging.
- We save **two** checkpoints per iteration: a numbered snapshot
  `model_iter_N.pt` (so you can compare or pit iterations against each other with
  `eval`, [Chapter 15](15-playing-evaluating-viewer.md)), and a rolling
  `latest.pt` that always points at the newest network (convenient for `play`).

### Reading the printed metrics

A real run prints something like:

```
Device: mps | sims/move: 100 | games/iter: 20
  [iter 0] self-play game 1/20: 143 positions
  [iter 0] self-play game 2/20: 88 positions
  ...
[iter 0] positions+2411 buffer=2411 policy_loss=6.9312 value_loss=0.9021 time=612.4s -> checkpoints/model_iter_1.pt
[iter 1] positions+2550 buffer=4961 policy_loss=6.1044 value_loss=0.8447 time=628.1s -> checkpoints/model_iter_2.pt
```

| Field | Meaning |
| --- | --- |
| `Device` / `sims/move` / `games/iter` | the resolved config for this run |
| `NNN positions` (per game) | number of `(state, π, z)` examples that game produced (≈ its length in plies) |
| `positions+N` | new examples added to the buffer this iteration |
| `buffer=N` | current buffer size (grows until it hits `replay_buffer_size`, then evicts oldest) |
| `policy_loss` | cross-entropy between the network policy and the MCTS policy π |
| `value_loss` | MSE between the predicted value and the game outcome z |
| `time` | wall-clock seconds for the iteration (self-play dominates) |

We interpret these numbers — and what healthy vs. unhealthy curves look like — in
[Chapter 16](16-debugging-and-convergence.md).

## 14.8 What a checkpoint is: `checkpoint.py`

A **checkpoint** is a saved snapshot of the model (and enough state to resume). It
lives in [`checkpoint.py`](../src/chesszero/checkpoint.py):

```python
def save_checkpoint(path, net, config, optimizer=None, iteration=0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "config": asdict(config),
        "iteration": iteration,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
```

The payload is a plain dictionary with four things:

| Key | What it is | Why we need it |
| --- | --- | --- |
| `model_state` | `net.state_dict()` — every weight and buffer in the network | the learned parameters themselves |
| `config` | the `Config` as a dict (`asdict`) | to rebuild the *same architecture* on load |
| `iteration` | which iteration produced this | to continue numbering on resume |
| `optimizer_state` | Adam's moment estimates (optional) | to resume training smoothly |

Loading reverses the process:

```python
def load_checkpoint(path, device=None):
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    config = Config(**payload["config"])
    if device:
        config.device = device
    net = ChessNet(config).to(config.device)
    net.load_state_dict(payload["model_state"])
    return net, config, payload
```

Note the order: we reconstruct the `Config`, build a fresh `ChessNet` with that
config (so the layer shapes match exactly), and only then load the weights into
it. Storing the config alongside the weights is what makes a checkpoint
self-describing.

### 📐 What is a `.pt` file, really?

`torch.save` serializes a Python object graph with `pickle` and writes it to disk;
the `.pt` extension is just convention (`.pth` is also common). It is **not** a
special binary format — it is a pickle containing tensors. That has a consequence:

> ⚠️ **Security:** we pass `weights_only=False` because our payload contains a
> `config` dict and an `iteration` int alongside the tensors, and `weights_only=True`
> (the modern default) would refuse to unpickle those. Unpickling arbitrary files
> can execute arbitrary code, so **only ever `load_checkpoint` files you produced
> or trust.** Never load a `.pt` file downloaded from a stranger with
> `weights_only=False`.

## 14.9 Running it

Putting it all together, a small end-to-end run that also saves games for the
viewer:

```bash
poetry run chesszero loop --iterations 8 --games 10 --simulations 100 --sample-games 10
```

This runs 8 iterations of (10 self-play games → train), writes
`checkpoints/model_iter_1.pt … model_iter_8.pt` plus `checkpoints/latest.pt`, and
drops 10 sampled games into `games/`. In [Chapter 15](15-playing-evaluating-viewer.md)
we will play against `latest.pt`, measure its strength with `eval`, and watch the
sampled games in the browser.

---

## Key takeaways

- `cmd_loop` is the code form of **generalized policy iteration**: repeat
  [self-play](12-self-play-in-code.md) (Phase A) then [training](13-training-and-replay-buffer.md)
  (Phase B), each iteration using a stronger network than the last.
- The network is **frozen** during self-play (`net.eval()`) and updated **only
  once per iteration** in `train_epochs` — not per game and not per move.
- `_evenly_spaced_indices` selects a small set of games spread across the entire
  run so the viewer can show learning over time; `_save_game` writes them as JSON.
- A **checkpoint** stores `model_state`, the `config`, the `iteration`, and
  (optionally) `optimizer_state`; `--resume` restores all of these — but **not**
  the replay buffer, which restarts empty.
- A `.pt` file is a `torch.save` pickle. We load with `weights_only=False`, so
  only load checkpoints you trust.

## Exercises

1. Trace one full iteration by hand: for `--iterations 2 --games 3`, what is
   `total_games`? With `--sample-games 4`, exactly which `global_index` values
   land in `sample_set`? (Use the formula in §14.5.)
2. In §14.6, why do we call `net.eval()` before self-play? What might go wrong if
   we forgot and left the network in `train()` mode during MCTS evaluation?
   (Hint: batch-norm statistics.)
3. Suppose you `--resume` a run that had a buffer of 40,000 examples. On the first
   resumed iteration, how many examples does `train_epochs` see? Why? What change
   would fix this (see [Chapter 16](16-debugging-and-convergence.md))?
4. `save_checkpoint` writes both `model_iter_N.pt` and `latest.pt` every
   iteration. Give one concrete use for the numbered files that `latest.pt` alone
   could not serve. (Peek at `eval` in [Chapter 15](15-playing-evaluating-viewer.md).)
5. Run `poetry run chesszero loop --iterations 2 --games 2 --simulations 20`.
   Watch the printed metrics. Did `value_loss` move? Why is two iterations far too
   few to conclude anything? (See [Chapter 16](16-debugging-and-convergence.md).)

---

> **Course:** [Home](README.md) · **Prev:** [13. Training & the Replay Buffer](13-training-and-replay-buffer.md) · **Next:** [15. Playing, Evaluating & the Viewer](15-playing-evaluating-viewer.md)
