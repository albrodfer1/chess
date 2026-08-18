# Chapter 16 — Debugging & Understanding Training

> **Course:** [Home](README.md) · **Prev:** [15. Playing, Evaluating & the Viewer](15-playing-evaluating-viewer.md) · **Next:** [17. Scaling Up & Improvements](17-scaling-and-improvements.md)

**What you'll learn**
- How to read every number `cmd_loop` prints, and what a *healthy* run looks like
- Why the loss going down does **not** prove your agent is getting stronger — and what does
- The four classic ways self-play training goes wrong, and the exact `Config` knob to turn for each
- A practical "symptom → cause → fix" table you can keep next to the terminal

---

## 16.1 The problem with training RL

When you train a supervised model, the loss curve tells you almost everything: down is good, flat is bad, up is a bug. Reinforcement learning is sneakier. Your loss can fall beautifully while your agent gets **worse**, or plateau while it gets **much stronger**. The reason is that the target is *moving*: the network is trained to imitate its own search (the policy target π) and to predict its own game outcomes (the value target z), and both of those change every iteration as the network changes.

So debugging ChessZero is less about staring at one curve and more about **triangulating** from several signals: the printed metrics, head-to-head evaluation games, and the browser viewer. This chapter teaches you to read all three, and then catalogues the failure modes you're most likely to hit.

Everything here refers to the real training loop in [`cli.py`](../src/chesszero/cli.py) (`cmd_loop`), the losses in [`train.py`](../src/chesszero/train.py), and the hyperparameters in [`config.py`](../src/chesszero/config.py).

## 16.2 Reading the output of `cmd_loop`

Run a short loop and you'll see something like this:

```
Device: mps | sims/move: 100 | games/iter: 20
  [iter 0] self-play game 1/20: 143 positions
  [iter 0] self-play game 2/20: 88 positions
  ...
[iter 0] positions+2011 buffer=2011 policy_loss=8.1043 value_loss=0.9614 time=642.7s -> checkpoints/model_iter_1.pt
[iter 1] positions+1876 buffer=3887 policy_loss=6.7215 value_loss=0.8977 time=610.2s -> checkpoints/model_iter_2.pt
```

Let's decode every field.

### The header line

```
Device: mps | sims/move: 100 | games/iter: 20
```

| Field | Source | Meaning |
| --- | --- | --- |
| `Device` | `config.device` | where tensors live: `cuda`, `mps` (Apple Silicon), or `cpu`. If this says `cpu` and you *have* a GPU, something is wrong — check [Chapter 17](17-scaling-and-improvements.md). |
| `sims/move` | `config.num_simulations` | how many MCTS simulations run per move. This is your search strength dial. |
| `games/iter` | `config.games_per_iteration` | how many self-play games feed each training phase. |

### The per-game lines

```
  [iter 0] self-play game 2/20: 88 positions
```

Each self-play game contributes one training **example per ply** (one board position it actually played through). "88 positions" means this game lasted 88 plies (44 full moves) before it ended. Watching these numbers is a cheap early-warning system:

- **Almost every game printing ~200 positions?** Your games are hitting the `max_moves` cap (default `200`) without a decisive result. That is the *all-draws collapse* — the single most common way training stalls (see §16.5).
- **Games ending in a wide spread of lengths (20, 60, 143, 88…)?** Healthy. Decisive games of varied length mean the value head is getting real win/loss signal.

### The per-iteration summary line

```
[iter 0] positions+2011 buffer=2011 policy_loss=8.1043 value_loss=0.9614 time=642.7s -> checkpoints/model_iter_1.pt
```

| Field | Meaning | What "healthy" looks like |
| --- | --- | --- |
| `positions+2011` | new examples added this iteration (`new_examples`) | grows with `games_per_iteration` × game length |
| `buffer=2011` | current size of the replay buffer | rises, then **plateaus** at `replay_buffer_size` (default `50_000`) as old data is evicted |
| `policy_loss` | mean cross-entropy between `policy_logits` and the MCTS policy π | starts near **8.4**, falls over many iterations |
| `value_loss` | mean MSE between the value head and outcomes z | starts near **1.0**, falls toward — but never reaching — 0 |
| `time` | wall-clock seconds for the whole iteration | dominated by self-play, i.e. `games/iter × plies × sims/move` network calls |
| `-> checkpoints/...` | where this iteration's checkpoint was written | one file per iteration, plus `latest.pt` |

📐 **Why policy_loss starts at ~8.4.** The policy head outputs a distribution over `ACTION_SIZE = 4672` moves. Before training, the network is random, so its softmax is roughly uniform. Cross-entropy against a target concentrated on a handful of legal moves is then about $\ln(\text{number of plausible moves})$. In the limit of a uniform distribution over all 4672 actions the cross-entropy approaches $\ln 4672 \approx 8.45$. So a fresh run *should* print a `policy_loss` in the low-to-mid 8s. If it starts far from there, your policy target or masking is suspect (revisit [Chapter 9](09-encoding-board-and-moves.md) and [Chapter 11](11-mcts-in-code.md)).

## 16.3 What healthy curves look like

```
policy_loss                         value_loss
8.4 ┤●                              1.0 ┤●
    │ ●                                 │ ●●
6   │   ●●                           0.7│   ●●●
    │     ●●●●                          │      ●●●●●
4   │        ●●●●●●●●                0.4│           ●●●●●●●
    │               ●●●●●●              │                 ●●●●
2   └────────────────────── iters    0.1└────────────────────── iters
```

- **`policy_loss` falls steadily** from ~8.4. This means the raw network is learning to *anticipate* what MCTS will decide — its "intuition" is catching up to its "calculation." A good sign the policy head is learning.
- **`value_loss` falls but flattens** well above zero. It can *never* reach zero: many positions are genuinely uncertain (a roughly equal middlegame really is a coin-flip), so predicting the outcome perfectly is impossible. A plateau around, say, 0.3–0.6 is normal, not a bug.

⚠️ **Down is necessary, not sufficient.** Both losses can fall while the agent gets *weaker*, if the network is overfitting to a shrinking, repetitive slice of self-play (see §16.6). Never declare victory from the loss alone. Confirm with §16.4.

## 16.4 Measuring *real* progress (not just loss)

The only trustworthy question is: **does a later network beat an earlier one?** You have two tools for answering it, both from [Chapter 15](15-playing-evaluating-viewer.md).

### Head-to-head evaluation

`cmd_eval` plays two checkpoints against each other (alternating colours so nobody gets a first-move edge), or a single checkpoint against a random mover:

```bash
# Is iteration 8 actually stronger than iteration 2?
poetry run chesszero eval checkpoints/model_iter_8.pt \
    --model-b checkpoints/model_iter_2.pt --games 20

# Sanity floor: can the current net crush a random player?
poetry run chesszero eval checkpoints/latest.pt --games 20
```

A network that is genuinely learning should, over many games:
1. **Beat a random player almost every game** within a few iterations. If it can't beat random after several iterations, something is fundamentally broken — start with §16.5.
2. **Beat its own earlier checkpoints** more often than it loses to them.

Because a single game is noisy, always use `--games 20` or more and look at the aggregate `model_a wins / model_b wins / draws` line that `cmd_eval` prints.

### Watching the viewer

Train with `--sample-games 10` so the loop saves games spread across the whole run ([Chapter 12](12-self-play-in-code.md) explains the capture; [Chapter 15](15-playing-evaluating-viewer.md) explains the UI), then:

```bash
poetry run chesszero viewer
```

Flip between an **early** game and a **late** game and watch the evaluation bars. Early on, the policy softmax is nearly flat — the network has no opinion, so probability mass is smeared across many moves. As training progresses, the mass should **concentrate** on a few sensible moves, and the value gauge should start reacting sharply to blunders and winning positions. Seeing that concentration happen is the most satisfying confirmation that learning is real.

## 16.5 Failure mode #1 — the all-draws collapse (the value signal dies)

This is the big one. If self-play games almost never end in a decisive result, the value target z is `0` for nearly every position, the value head learns to just output ~0 everywhere, and learning grinds to a halt.

**How to spot it:** the per-game lines nearly all print `~200 positions` (hitting `max_moves`), and `value_loss` sits stuck at a low-but-flat value because predicting "always 0" is easy and always roughly right.

📐 **A crucial distinction — quality vs. signal.** It's tempting to blame "too few simulations," but be precise about the mechanism:

- **Low `num_simulations`** makes each *move* weaker and noisier, but thanks to **bootstrapping** (a non-terminal leaf is scored by the value head, not by playing to the end — see [Chapter 6](06-monte-carlo-tree-search.md) and [Chapter 11](11-mcts-in-code.md)) MCTS *still produces a value* for the position. Low sims degrade **move quality**.
- **Games never ending decisively** is different: it starves the *training target* of ground truth. The real win/loss signal enters the system **only** through completed games ([Chapter 5](05-self-play-and-games.md)). No decisive games → no learning signal, no matter how good your search is.

**Fixes, in order of leverage:**

1. **Raise `max_moves`** so a winning side has time to actually deliver mate instead of getting truncated to a draw. (But see the caveat below — early on this can just make slow draws slower.)
2. **Raise `num_simulations`** so play is purposeful enough to convert advantages into wins rather than shuffling aimlessly.
3. **Be patient through the cold-start** (§16.7). A brand-new random network draws or stalemates a lot; that's expected for the first iteration or two, not forever.

⚠️ Raising `max_moves` alone, with a still-clueless network, can *increase* your wall-clock time without producing more decisive games — you just get longer aimless games. The durable fix is a network strong enough to win won positions, which comes from sims + iterations together.

## 16.6 Failure mode #2 — overfitting to itself / catastrophic forgetting

Self-play data is generated by the *current* network, so it reflects the current network's habits. If the network narrows into one repetitive style, it trains on an ever-shrinking slice of chess, overfits to it, and **forgets** how to handle positions it no longer visits. Strength can then oscillate or regress even as loss falls.

**Defenses (some built in, some to add):**

- **The replay buffer is your main defense.** `ReplayBuffer` in [`replay_buffer.py`](../src/chesszero/replay_buffer.py) is a `deque(maxlen=replay_buffer_size)`. Training samples a random minibatch from the *last* `replay_buffer_size` examples ([Chapter 13](13-training-and-replay-buffer.md)), so each update sees a mix of recent games, not just the latest one. A **larger** `replay_buffer_size` remembers more history and resists forgetting; too small and you overfit to the newest games.
- **Keep every checkpoint.** `cmd_loop` writes `model_iter_N.pt` each iteration (not just `latest.pt`). This lets you use `cmd_eval` to detect regression — if `model_iter_9` loses to `model_iter_6`, you have evidence of forgetting and can roll back.
- **Arena gating** (an improvement, not yet in the code): only *promote* a new network to generate the next batch of self-play if it actually beats the current best in evaluation. This is exactly how AlphaGo Zero avoided regressions. See [Chapter 17](17-scaling-and-improvements.md) for where this would slot into the loop.

## 16.7 Failure mode #3 — insufficient exploration

If self-play is too deterministic, every game looks the same, the dataset lacks variety, and the network never discovers better moves because it never *tries* them. The two exploration knobs live in `config.py`:

- **Dirichlet noise at the MCTS root** — `dirichlet_alpha` (default `0.3`) and `dirichlet_epsilon` (default `0.25`). This noise is mixed into the root priors in `_add_dirichlet_noise` ([`mcts.py`](../src/chesszero/mcts.py)) *only during self-play* (`add_noise=True`), nudging the search to occasionally explore moves the network currently underrates. Lower `dirichlet_epsilon` → less exploration (more exploitation of current beliefs); higher → more random probing. `dirichlet_alpha` shapes *how* that noise spreads across moves (smaller α → spikier, concentrated on a few moves).
- **Temperature on move selection** — `temperature_moves` (default `30`). For the first `temperature_moves` plies of each self-play game, `select_move` samples a move in proportion to visit counts (temperature 1.0) instead of always taking the most-visited move (temperature 0.0). This diversifies openings ([Chapter 12](12-self-play-in-code.md)). Set it to `0` and every game starts identically; raise it and games branch out earlier.

**Symptom of too little exploration:** in the viewer, games from different iterations follow near-identical opening lines, and `positions` per game clusters tightly. **Fix:** raise `dirichlet_epsilon` and/or `temperature_moves`.

⚠️ There is such a thing as *too much* exploration: crank the noise up and self-play becomes near-random, producing low-quality games that teach the network little. Exploration is a dial to balance, not to max out.

## 16.8 Failure mode #4 — the cold-start (be patient)

At iteration 0 the network weights are random ([Chapter 5](05-self-play-and-games.md)). Its policy is ~uniform and its value output is noise. The *only* real information in the whole system at that point comes from games that happen to reach a decisive terminal state — a lucky checkmate that MCTS stumbles into via `terminal_value`. From that thin signal the value head slowly grounds, which makes the policy target better, which makes the next batch of games better.

**Implication:** do not panic if the first iteration or two look terrible — near-random play, lots of draws, high loss. That is the bootstrap starting from nothing. Judge the run over *many* iterations, and use `cmd_eval` vs. random as your "are we above the floor yet?" check.

## 16.9 Practical, mechanical bugs

Not every problem is deep RL dynamics. Some are ordinary software bugs:

- **Device mismatches.** Everything must live on `config.device`. The code already handles this (`Evaluator` moves inputs with `.to(self.device)`; `_build` moves the net with `.to(config.device)`), but if you extend the code and create a tensor without moving it, you'll get a `cpu`/`mps`/`cuda` mismatch error. `default_device()` in `config.py` picks `cuda → mps → cpu`; force one explicitly with the global `--device` flag if auto-detection misbehaves.
- **NaN or exploding loss.** If `policy_loss` or `value_loss` prints `nan` or shoots upward, your `learning_rate` (default `1e-3`) is likely too high for your setup. Lower it (e.g. `5e-4`). NaNs can also come from a degenerate batch; confirm the buffer actually has data (`buffer=` should be non-zero).
- **The replay buffer is in-memory only.** `ReplayBuffer` lives in RAM. `cmd_loop` never calls its `save`/`load` methods, and `--resume` restores only the **network and optimizer** from the checkpoint — **not** the buffer. So a resumed run starts with an empty buffer and its first training phase sees only freshly generated games. This isn't a crash, but it's a silent discontinuity that can dent strength right after a resume. ([Chapter 17](17-scaling-and-improvements.md) shows how to persist the buffer using the existing `save`/`load` methods.)
- **Checkpoint loads with `weights_only=False`.** `load_checkpoint` ([Chapter 14](14-the-reinforcement-loop.md)) deserializes a full pickle, which is fine for your own files but means you should never load a `.pt` from an untrusted source.

## 16.10 Symptom → cause → fix

| Symptom | Likely cause | Knob / action |
| --- | --- | --- |
| Almost every game prints `~200 positions` | Games truncating at the move cap; value signal dying | Raise `max_moves`; raise `num_simulations`; wait out the cold-start |
| `value_loss` stuck flat and low from iteration 0 | Value head learned "always 0" because almost all z = 0 | Same as above — get decisive games |
| `policy_loss` never drops below ~8 | Policy target/masking broken; net not learning intuition | Recheck `move_to_index`/`legal_mask` ([Ch. 9](09-encoding-board-and-moves.md)) and the CE target ([Ch. 13](13-training-and-replay-buffer.md)) |
| Loss falls but `eval` shows no strength gain | Overfitting to self / forgetting | Increase `replay_buffer_size`; keep checkpoints; add arena gating |
| Every iteration's games look identical in the viewer | Too little exploration | Raise `dirichlet_epsilon`, `dirichlet_alpha`, `temperature_moves` |
| Games look random and low-quality | Too *much* exploration | Lower `dirichlet_epsilon` / `temperature_moves` |
| First 1–2 iterations look awful | Cold-start (random net) | Normal — judge over many iterations |
| `loss = nan` or exploding | Learning rate too high | Lower `learning_rate` (e.g. `1e-3 → 5e-4`) |
| Runs on `cpu` despite having a GPU | Device auto-detect failed | Pass `--device cuda`/`mps`; check `default_device()` |
| Strength dips right after `--resume` | Replay buffer not restored | Persist/reload the buffer ([Ch. 17](17-scaling-and-improvements.md)) |
| Iterations take forever | Self-play cost = games × plies × sims | Lower `num_simulations`/`games_per_iteration`; use a GPU ([Ch. 17](17-scaling-and-improvements.md)) |

## 16.11 A minimal debugging workflow

When a run misbehaves, work from cheap checks to expensive ones:

1. **Glance at the per-game `positions`.** All ~200? You have the all-draws collapse — go to §16.5.
2. **Glance at the loss line.** Starts near 8.4 for policy and ~1.0 for value, then both trend down? Encoding and training are wired correctly.
3. **Run `eval` vs. random.** Can't beat random after a few iterations? Something is broken, not just slow — recheck the pipeline end to end.
4. **Open the viewer.** Compare early vs. late games; watch whether policy mass concentrates. This tells you *how* the agent is changing, not just *whether*.
5. **Only then reach for knobs.** Change **one** hyperparameter at a time so you can attribute the effect.

---

## Key takeaways

- `cmd_loop` prints, per iteration, the new/total example counts and the mean `policy_loss` and `value_loss`; `policy_loss` starts near `ln 4672 ≈ 8.4` and both should trend down.
- **Falling loss is necessary but not sufficient.** Confirm real strength with `cmd_eval` (head-to-head, many games) and by watching policy mass concentrate in the viewer.
- The dominant failure is the **all-draws collapse**: no decisive games → no value signal. Distinguish it from low simulations, which only hurt move *quality* thanks to bootstrapping.
- Guard against **forgetting** with a large replay buffer, retained checkpoints, and (as an upgrade) arena gating; tune **exploration** with `dirichlet_*` and `temperature_moves`.
- Remember the mechanical gotchas: device placement, NaNs from a too-high learning rate, and the **in-memory buffer that `--resume` does not restore**.

## Exercises

1. Start a fresh run and record the very first `policy_loss` it prints. Is it close to 8.4? Compute `ln(4672)` and explain any gap in terms of how many moves the initial policy spreads mass over.
2. Deliberately induce the all-draws collapse: set `max_moves` very low (e.g. 10) and run a few iterations with `--sample-games`. What happens to `value_loss` and to the per-game `positions`? Explain using §16.5.
3. Train two short runs identical except for `temperature_moves` (say `0` vs. `30`). Open both in the viewer. How does the opening variety differ? Which knob did you change and why did it matter?
4. Using `cmd_eval`, design an experiment to detect *forgetting*: which checkpoints would you pit against each other, how many games, and what result would count as evidence of regression?
5. You `--resume` a run and notice strength dips for one iteration then recovers. Trace the cause through §16.9 and propose the fix you'll read about in [Chapter 17](17-scaling-and-improvements.md).

---

> **Course:** [Home](README.md) · **Prev:** [15. Playing, Evaluating & the Viewer](15-playing-evaluating-viewer.md) · **Next:** [17. Scaling Up & Improvements](17-scaling-and-improvements.md)
