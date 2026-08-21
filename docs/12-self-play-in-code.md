# Chapter 12 — Self-Play

> **Course:** [Home](README.md) · **Prev:** [11. MCTS in Code](11-mcts-in-code.md) · **Next:** [13. Training & the Replay Buffer](13-training-and-replay-buffer.md)

**What you'll learn**
- How one network plays *both* sides of a game to generate its own training data
- The exact life of a single self-play game in [`selfplay.py`](../src/chesszero/selfplay.py), move by move
- Why MCTS runs on *every* move but the reward only arrives at the *end* — and how we assign that reward backward to every position
- The temperature schedule that balances exploration and strength
- Why games are *guaranteed* to end, and the `record=True` path that feeds the viewer
- How **batched self-play** runs many games at once so one GPU forward pass evaluates many positions

---

## 12.1 Where we are

This is **Pillar 2** of our agent (see [Chapter 1](01-introduction.md)): *the model
learns by playing itself.* We now have all the pieces to make that concrete:

- [Chapter 5](05-self-play-and-games.md) argued *why* self-play works — the reward
  is ground truth, search is a policy-improvement operator, and the opponent is
  always exactly your own strength (an automatic curriculum).
- [Chapter 9](09-encoding-board-and-moves.md) gave us `encode_board` to turn a
  position into a tensor.
- [Chapter 11](11-mcts-in-code.md) gave us `run_mcts`, `policy_from_visits`, and
  `select_move` — the search that turns the network's shallow intuition into a
  sharpened policy.

This chapter assembles them into `play_game`: the function that produces the
`(state, policy, value)` training examples that [Chapter 13](13-training-and-replay-buffer.md)
will learn from.

The whole file is barely 120 lines. That is the point — self-play is
conceptually simple. The subtlety is all in *what gets recorded when*.

## 12.2 One network, two players

The single most important idea in this chapter is that **there is only one
network**. It does not play "White's network" against "Black's network." The
same weights choose moves for both sides:

```python
board = chess.Board()
...
while not board.is_game_over() and move_number < config.max_moves:
    root = run_mcts(board, evaluator, config, add_noise=True)
    ...
    move = select_move(root, temperature=temperature)
    board.push(move)
    move_number += 1
```

Look closely: the loop just keeps pushing moves onto one `board`. On White's
turn, `board.turn` is White and the search reasons from White's perspective; on
Black's turn it reasons from Black's. The `Evaluator` (from
[Chapter 11](11-mcts-in-code.md)) always returns a value *from the side-to-move's
point of view*, so the same network naturally plays both colours without any
special casing.

This is what makes self-play a closed loop with no external opponent: the agent
is its own sparring partner, and — as [Chapter 5](05-self-play-and-games.md)
explained — that sparring partner improves in lockstep with the agent itself.

## 12.3 The anatomy of one game

Here is `play_game` with the recording path removed, so you can see the essential
skeleton:

```python
def play_game(evaluator, config, record=False):
    board = chess.Board()
    history = []  # (state, policy, side_to_move)

    move_number = 0
    while not board.is_game_over() and move_number < config.max_moves:
        root = run_mcts(board, evaluator, config, add_noise=True)   # 1. THINK
        if not root.children:
            break

        temperature = 1.0 if move_number < config.temperature_moves else 0.0
        policy = policy_from_visits(root, temperature=1.0)          # 2. RECORD π
        history.append((encode_board(board), policy, board.turn))   # 3. STORE

        move = select_move(root, temperature=temperature)           # 4. PICK
        board.push(move)                                            # 5. PLAY
        move_number += 1

    result = _game_result(board)                                    # 6. WHO WON?

    examples = []
    for state, policy, side_to_move in history:                     # 7. LABEL
        value = result if side_to_move == chess.WHITE else -result
        examples.append(Example(state=state, policy=policy, value=float(value)))
    return examples
```

Let's walk the seven numbered steps.

### Step 1 — Think (run MCTS)

```python
root = run_mcts(board, evaluator, config, add_noise=True)
```

For the *current* position, run a full Monte Carlo Tree Search — by default 100
simulations (`config.num_simulations`). Note `add_noise=True`: this injects
**Dirichlet noise** at the root ([Chapter 6](06-monte-carlo-tree-search.md),
[Chapter 11](11-mcts-in-code.md)), nudging the search to occasionally explore
moves the network currently underrates. Exploration is *essential* here — without
it, self-play would play the same handful of games forever and learn almost
nothing. During real play ([Chapter 15](15-playing-evaluating-viewer.md)) noise is
turned *off*; it exists purely to diversify training data.

⚠️ **MCTS runs on *every single move*, not once at the end of the game.** This
trips up almost everyone the first time. Each move is a fresh 100-simulation
search from the position in front of us. A 60-move game therefore runs ~120
searches (one per ply). The end of the game contributes only the final label `z`
(step 6) — it does *not* "run the search."

If the search finds no legal moves (`not root.children`), the game is already
effectively over and we break out.

### Step 2 & 3 — Record the search policy and store the position

```python
policy = policy_from_visits(root, temperature=1.0)
history.append((encode_board(board), policy, board.turn))
```

MCTS produces an **improved policy** `π`: a probability distribution over moves
proportional to how many times each move was visited during the search
([Chapter 11](11-mcts-in-code.md), `policy_from_visits`). This is the *training
target* for the policy head — the network will later be taught to imitate it
([Chapter 13](13-training-and-replay-buffer.md)).

We store three things for this position:

| Stored | What it is | Becomes… |
| --- | --- | --- |
| `encode_board(board)` | the (19, 8, 8) input tensor ([Ch. 9](09-encoding-board-and-moves.md)) | the network **input** |
| `policy` (`π`) | visit-count distribution over 4672 actions | the **policy target** |
| `board.turn` | whose move it is (White/Black) | used to sign the **value target** in step 7 |

📐 **A subtlety worth pausing on: the recorded `π` always uses `temperature=1.0`,
even late in the game.** The move we *play* may be chosen greedily (step 4), but
the target we *train on* is always the full visit-proportional distribution. Why?
Because a sharp, honest distribution of "how much did search prefer each move" is
a richer learning signal than a one-hot "it played this." Selection temperature
and target temperature are decoupled on purpose.

### Step 4 — Pick a move (the temperature schedule)

```python
temperature = 1.0 if move_number < config.temperature_moves else 0.0
move = select_move(root, temperature=temperature)
```

Now we choose the move to actually play. This uses a **temperature schedule**
(`config.temperature_moves` defaults to 30):

- **First 30 plies — `temperature = 1.0`:** sample a move *in proportion to its
  visit count*. Sometimes the second- or third-best move gets played. This
  diversifies openings so the network sees a wide variety of positions.
- **After 30 plies — `temperature = 0.0`:** play *greedily*, the single
  most-visited move. Once the game is under way we want the strongest play so the
  eventual result is a meaningful signal.

This is the exploration/exploitation trade-off from [Chapter 2](02-rl-fundamentals.md)
in action: explore early (when variety matters), exploit late (when strength
matters). Together with the Dirichlet noise of step 1, it is what keeps self-play
games from collapsing into a single repeated line.

### Step 5 — Play it

```python
board.push(move)
move_number += 1
```

Apply the move. The board advances, `board.turn` flips to the other side, and the
loop repeats — now searching for the opponent (which is, again, the same
network).

### Step 6 — Who won?

```python
result = _game_result(board)
```

When the loop ends, we compute the outcome:

```python
def _game_result(board):
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0
```

`result` is `+1` (White won), `-1` (Black won), or `0` (draw *or* the game hit the
move limit without a decision). This single number is the only **reward** the
whole system ever receives — the ground-truth signal of [Chapter 5](05-self-play-and-games.md).

### Step 7 — Label every position (credit assignment)

```python
for state, policy, side_to_move in history:
    value = result if side_to_move == chess.WHITE else -result
    examples.append(Example(state=state, policy=policy, value=float(value)))
```

This is the heart of self-play learning, and it is where the **credit-assignment
problem** from [Chapter 1](01-introduction.md) is solved with beautiful crudeness.

We do not try to figure out *which* move won the game. We simply say: **every
position that side eventually played from is labelled with the game's final
result, from that side's perspective.**

- A position where it was White's move, in a game White won → `value = +1`.
- A position where it was Black's move, in that same game → `value = -1`.
- Every position in a drawn game → `value = 0`.

This is the **Monte Carlo return** of [Chapter 2](02-rl-fundamentals.md): with no
discounting and a reward only at the end, the return of *every* state in the
episode is just the final outcome. The sign flip (`result` vs `-result`) enforces
the side-to-move convention from [Chapter 5](05-self-play-and-games.md): a value
of `+1` always means "good for whoever is about to move here."

Is this noisy? Absolutely — a brilliant move in a lost game still gets labelled
`-1`. But *averaged over thousands of games*, positions that genuinely tend to
lead to wins accumulate positive labels and those that lead to losses accumulate
negative ones. The noise cancels; the signal survives. This is exactly the
policy-evaluation half of the policy-iteration story from
[Chapter 7](07-the-alphazero-algorithm.md).

### One game → many examples

```
        one self-play game (say 40 plies)
   ┌──────────────────────────────────────────────┐
   move 0   move 1   move 2   ...            move 39
   (White)  (Black)  (White)                 (Black)
     │        │        │                        │
     ▼        ▼        ▼                        ▼
  (s0,π0,W)(s1,π1,B)(s2,π2,W) ...          (s39,π39,B)
     │        │        │                        │
     └────────┴────────┴─── result z = +1 ──────┘   (White won)
              │        │                        │
   label:   +1       -1       +1      ...      -1
     (z if White to move, -z if Black to move at that position)

   40 positions  →  40 Example(state, π, value)  →  ReplayBuffer
```

Every game a single network plays against itself becomes a batch of labelled
training examples. Multiply by `games_per_iteration` (default 20) and you have a
few thousand examples per iteration — the fuel for [Chapter 13](13-training-and-replay-buffer.md).

## 12.4 Games are guaranteed to end

A subtle worry: what if two weak networks shuffle pieces forever and no game ever
finishes? Then no `result` is ever produced, and no learning happens. The code
closes this hole two ways.

**1. The rules themselves force termination.** `board.is_game_over()` becomes
`True` not only on checkmate and stalemate, but on the *automatic* draw rules that
`python-chess` enforces without anyone claiming them:

- **Fivefold repetition** — the same position five times.
- **Seventy-five-move rule** — 75 moves by each side with no capture or pawn move.

Even two networks playing pure nonsense will eventually trip one of these. A game
*cannot* run forever.

**2. A hard move cap as a backstop.** The loop condition also checks
`move_number < config.max_moves` (default 200 plies):

```python
while not board.is_game_over() and move_number < config.max_moves:
```

If a game somehow reaches 200 plies, we stop and `_game_result` returns `0.0`
(treated as a draw, since `board.outcome()` is `None`).

⚠️ **The real risk is not *non-termination* but *all-draws*.** If nearly every
game ends `0`, the value target is almost always `0` and the value head learns
nothing useful. That failure mode — and how the temperature/noise exploration and
sufficient `num_simulations` guard against it — is discussed in
[Chapter 16](16-debugging-and-convergence.md). Termination is guaranteed;
*decisive* termination is what you tune for.

## 12.5 The `record=True` path — data for the viewer

Everything above is what training needs. But we also want to *watch* games in the
browser viewer ([Chapter 15](15-playing-evaluating-viewer.md)), and for that we
capture much richer per-move data. Passing `record=True` makes `play_game` return
`(examples, game_record)` instead of just `examples`.

For each ply, `_record_ply` builds a JSON-friendly dict:

```python
def _record_ply(board, root, move, evaluator, ply):
    priors, net_value = evaluator.evaluate(board)          # a CLEAN re-evaluation
    total_visits = sum(c.visit_count for c in root.children.values()) or 1

    evaluations = []
    for mv, child in root.children.items():
        evaluations.append({
            "uci": mv.uci(),
            "san": board.san(mv),
            "policy": round(priors.get(mv, 0.0), 5),               # network softmax
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
```

A few design points worth understanding:

- **Why re-evaluate?** During self-play the root priors were polluted with
  Dirichlet noise (step 1). To show the viewer the network's *honest* opinion, we
  call `evaluator.evaluate(board)` again to get a clean policy softmax and value,
  untouched by exploration noise. The comment in the code says exactly this.
- **Two numbers per move.** `policy` is the network's raw softmax probability for
  that move (its instant intuition); `visits` is the fraction of MCTS visits that
  move received (search's refined opinion). The viewer draws both, so you can
  literally *see* search sharpening intuition — the theme of
  [Chapter 7](07-the-alphazero-algorithm.md).
- **Two values per move.** `value` is the network's instant value-head estimate;
  `mcts_value` is `root.value`, the search-averaged value. Comparing them shows
  how much thinking changed the assessment.
- **Only the top 12 moves** are kept, sorted by network probability, to keep the
  files small.

After the loop, one extra "terminal" entry records the final board so the viewer
can display the finished position:

```python
ply_records.append({
    "ply": move_number, "fen": board.fen(),
    "turn": ..., "played": None, "value": None,
    "mcts_value": None, "evaluations": [], "terminal": True,
})
```

and the whole thing is wrapped with metadata:

```python
game_record = {
    "result": result,                          # +1 / 0 / -1 (white perspective)
    "result_str": board.result(claim_draw=True),  # "1-0", "0-1", "1/2-1/2"
    "num_plies": move_number,
    "termination": _termination_str(board),    # "checkmate", "stalemate", "move-limit", ...
    "moves": ply_records,
}
```

⚠️ **Recording is not free** — `_record_ply` does an extra network evaluation per
move and calls `board.san(...)` for every legal move. That is why full games are
recorded only for a *sparsely sampled* subset (10 by default, spread across
training), not for every game. The sampling logic lives in the training loop and
is covered in [Chapter 14](14-the-reinforcement-loop.md).

## 12.6 Generating many games at once (batched self-play)

Everything so far played **one game at a time**, and inside each game MCTS
evaluates **one position per simulation** — `Evaluator.evaluate` does an
`unsqueeze(0)`, a batch of size *one*. That's fine on a CPU, but it leaves a GPU
almost entirely idle: a GPU is built to run hundreds of positions in parallel, and
a batch of one is dominated by the fixed cost of launching the work and copying the
result back (a GPU↔CPU sync *every* simulation). Since **self-play is the training
bottleneck** ([Chapter 17](17-scaling-and-improvements.md)), this is the first
thing worth speeding up.

The fix is to **run many games concurrently and evaluate their positions
together**. It comes in three layers, bottom to top.

### Layer 1 — a batched evaluator

`Evaluator.evaluate_many` scores a *list* of boards in a single forward pass, and
`evaluate` becomes a one-element wrapper around it:

```python
@torch.no_grad()
def evaluate_many(self, boards):
    arr = np.stack([encode_board(b) for b in boards])   # (N, planes, 8, 8)
    x = torch.from_numpy(arr).to(self.device)
    logits, values = self.net(x)                        # ONE forward pass for all N
    ...                                                 # per-board legal-move softmax
    return results                                      # list of ({move: prior}, value)

def evaluate(self, board):
    return self.evaluate_many([board])[0]
```

One `self.net(x)` call now covers `N` positions instead of one — the difference
between a few percent GPU utilization and a full device.

### Layer 2 — batched search across trees

`run_mcts_batch` runs an **independent MCTS for each board**, but steps them in
lockstep so their evaluations can be pooled. Each simulation descends *every* tree
to a leaf, collects those leaves, and scores them all with one `evaluate_many`
call:

```python
for _ in range(config.num_simulations):
    leaves, paths = [], []
    for root in roots:                       # descend EACH tree to its leaf
        node, path = root, [root]
        while node.expanded:
            node = _select_child(node, config.c_puct); path.append(node)
        ...
        leaves.append(node); paths.append(path)

    pending = [n.board for n, term in zip(leaves, terminal) if not term]
    evals = iter(evaluator.evaluate_many(pending))   # ONE batched call for all trees
    ...                                              # expand + back up each tree as before
```

The trees never interact — the *only* thing shared is the batched network call.
`run_mcts` is now simply `run_mcts_batch([board])[0]`, so there is a single code
path.

📐 **This changes nothing about the search itself.** Batching only groups the
neural-network calls; each game's PUCT selection, expansion, and backup are exactly
what [Chapter 11](11-mcts-in-code.md) described. With Dirichlet noise off,
`run_mcts_batch([b, b])` produces two *identical* trees, bit-for-bit equal to
`run_mcts(b)` — this is checked in the tests.

### Layer 3 — a refill pool of games

`play_games_batch` drives whole games. The wrinkle is that **chess games vary
enormously in length** — a quick mate in 20 plies next to a 200-ply grind (you saw
exactly this in the training logs). If we launched a fixed block of games and waited
for all of them, the batch would collapse to a single straggler at the end, wasting
the GPU. So instead we keep a **pool** of `selfplay_batch_size` games in flight and
start a fresh game the instant one finishes:

```
 selfplay_batch_size = 4,  num_games = 8

 slot ┌ g0───────────done → g4────────done → g7──────done
      │ g1──done → g5──────────────────────done
      │ g2───────────────done → g6──────done
      └ g3──done → (pool drains as the last games finish)
        └────────── always ~4 games searching at once ──────────┘
```

Each outer step runs one batched search over the *active* games' current positions,
plays one ply in each, finalizes any that ended, and refills the empty slots.
Results come back in game-start order; an optional `on_game_done` callback fires as
each game finishes so the training loop can log live.

### Using it

The knob is `config.selfplay_batch_size` (default 16), exposed on the training loop
as `--parallel-games`:

```bash
chesszero --device cuda loop --parallel-games 32 --simulations 200 ...
```

Bigger batches use the GPU better, up to the point where you run out of memory or
active games. Even on CPU it's roughly **2× faster** (it amortizes Python/PyTorch
dispatch overhead); on a GPU the speedup is far larger, which is the whole point.

⚠️ **Two behavioural notes.** (1) Because games run concurrently and finish at
different times, the per-game log lines now appear in **finish order**, not start
order — the `game k/N` label is the start index. (2) Dirichlet noise is drawn in a
different order than sequential play, so a batched run isn't bit-identical to a
sequential one — but it's the same algorithm sampling from the same distribution,
so the training data is equivalent.

This is improvement **(b)** from [Chapter 17](17-scaling-and-improvements.md), now
built in.

## 12.7 How examples flow onward

`play_game` returns a list of `Example` objects (defined in
[`replay_buffer.py`](../src/chesszero/replay_buffer.py)):

```python
@dataclass
class Example:
    state: np.ndarray   # (19, 8, 8) float32   -> network input
    policy: np.ndarray  # (4672,) float32       -> policy target π
    value: float        # game outcome ∈ {-1,0,1} -> value target z
```

The training loop ([Chapter 14](14-the-reinforcement-loop.md)) drops these into a
`ReplayBuffer`, and the trainer ([Chapter 13](13-training-and-replay-buffer.md))
samples minibatches from it to fit the network. The `state` becomes the input,
`policy` is the target for the policy head, and `value` is the target for the
value head — precisely the two training signals of
[Chapter 7](07-the-alphazero-algorithm.md).

You now understand where every piece of training data comes from. Next we'll see
what the network *does* with it.

---

## Key takeaways

- **Self-play uses one network for both sides.** The `Evaluator`'s side-to-move
  value convention makes this automatic — no separate opponent exists.
- **`play_game` runs MCTS on every move**, records the visit-count policy `π` and
  the position, then plays a move under a **temperature schedule** (explore for
  the first `temperature_moves` plies, then play greedily).
- **The reward arrives only at the end.** Every stored position is then labelled
  with the final result from its own mover's perspective — the Monte Carlo return
  and the crude-but-effective solution to credit assignment.
- **Games always terminate** thanks to fivefold-repetition, the seventy-five-move
  rule, and a `max_moves` backstop. The thing to tune for is *decisive* games, not
  termination.
- **`record=True`** captures rich per-ply data (clean policy softmax, network vs
  MCTS value, top moves) for the browser viewer — at extra cost, so only sampled
  games are recorded.
- **Batched self-play** (`evaluate_many` → `run_mcts_batch` → `play_games_batch`)
  runs many games concurrently and pools their network calls into single forward
  passes, with a refill pool keeping the batch full despite varying game lengths.
  It's the main lever for using a GPU (`--parallel-games`) and doesn't change *what*
  is learned — only how fast games are generated.

## Exercises

1. A 50-ply game is played and White wins. How many `Example` objects does
   `play_game` produce, and what `value` does the position after move 10 (Black to
   move) get?
2. The recorded target `π` uses `temperature=1.0` even for a move played greedily
   with `temperature=0.0`. Explain in one sentence why training on the full
   distribution is better than training on the single played move.
3. Suppose you set `config.temperature_moves = 0`. What changes about the *variety*
   of self-play games, and why might that hurt learning? (Hint: revisit
   exploration in [Chapter 2](02-rl-fundamentals.md).)
4. Why does `_record_ply` call `evaluator.evaluate(board)` again instead of reading
   the priors already stored on the root's children? (Hint: `add_noise=True`.)
5. Trace what happens if a game reaches `config.max_moves`. What is `result`, what
   is `board.outcome()`, and what `value` do all the positions receive?
6. In `play_games_batch`, why keep a **refill pool** of games instead of running a
   fixed block of `selfplay_batch_size` games and waiting for all to finish? (Hint:
   look at the spread of game lengths in the training logs.) What happens to GPU
   utilization at the end of a fixed block?

---

> **Course:** [Home](README.md) · **Prev:** [11. MCTS in Code](11-mcts-in-code.md) · **Next:** [13. Training & the Replay Buffer](13-training-and-replay-buffer.md)
