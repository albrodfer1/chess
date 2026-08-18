# Chapter 13 — Training & the Replay Buffer

> **Course:** [Home](README.md) · **Prev:** [12. Self-Play](12-self-play-in-code.md) · **Next:** [14. The Reinforcement Loop & Checkpoints](14-the-reinforcement-loop.md)

**What you'll learn**
- How self-play examples are stored, mixed, and sampled in a **replay buffer**
- Exactly how the two-headed loss is computed in [`train.py`](../src/chesszero/train.py)
- Why the value head learns from a **reward** while the policy head learns from a **distilled search policy**
- How the code's loss maps, term for term, onto the AlphaZero loss from [Chapter 7](07-the-alphazero-algorithm.md)
- A real subtlety: cross-entropy over all 4672 actions, including illegal ones

---

## 13.1 Where we are

[Chapter 12](12-self-play-in-code.md) produced a stream of `Example` objects —
each a `(state, policy, value)` triple. This chapter is **Phase B** of the loop
from [Chapter 1](01-introduction.md): turning those examples into a better
network. Two small files do the work:

- [`replay_buffer.py`](../src/chesszero/replay_buffer.py) — stores examples and
  hands out random minibatches.
- [`train.py`](../src/chesszero/train.py) — runs gradient descent on the two-headed
  loss.

This is where the abstract idea "train the network toward search and outcomes"
([Chapter 7](07-the-alphazero-algorithm.md)) becomes concrete PyTorch.

## 13.2 The `Example`: one training row

Every position self-play produces is a single dataclass:

```python
@dataclass
class Example:
    state: np.ndarray   # (19, 8, 8) float32   -> the network INPUT
    policy: np.ndarray  # (4672,) float32       -> the policy TARGET  (π)
    value: float        # game outcome ∈ {-1,0,1} -> the value TARGET  (z)
```

Line these up with the network from [Chapter 10](10-the-neural-network.md):
`state` goes *in*; `policy` is what the policy head should output; `value` is what
the value head should output. Training is nothing more than pushing the network's
two outputs toward these two targets, over and over.

Recall from [Chapter 12](12-self-play-in-code.md) where the targets come from:

- `state` — `encode_board(board)`, the 19-plane tensor ([Ch. 9](09-encoding-board-and-moves.md)).
- `policy` (`π`) — MCTS visit-count distribution, the **improved search policy**.
- `value` (`z`) — the game's final result, signed for the side to move: the
  **Monte Carlo return** and the only true reward in the system.

## 13.3 The replay buffer

### What it is

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)   # default capacity = 50,000

    def add(self, examples):
        self.buffer.extend(examples)
```

A `deque` with a fixed `maxlen` is a **sliding window over recent experience**.
When it is full and you add more, the oldest examples fall off the left end
automatically. So the buffer always holds roughly the last `replay_buffer_size`
positions the agent generated — a mix of games from the last several iterations,
not just the most recent one.

### Why bother? Three reasons

The buffer is not just a container; it is a training-stability tool
([Chapter 4](04-deep-rl-and-function-approximation.md)):

1. **Decorrelation.** Consecutive positions *within a game* are highly correlated
   (the board barely changes move to move). Training on them in order would give
   the optimizer a biased, jittery gradient. Sampling *randomly* from a large
   buffer breaks that correlation, approximating the i.i.d. data that
   stochastic gradient descent assumes.
2. **Data reuse.** Self-play is expensive (100 MCTS simulations per move!). Each
   example is reused across several epochs and iterations instead of being seen
   once and thrown away.
3. **Mitigating catastrophic forgetting.** Because the buffer retains games from
   *earlier* iterations, the network keeps being reminded of positions it has
   stopped playing, so it does not overfit to its current pet lines and forget how
   to handle everything else (more in [Chapter 16](16-debugging-and-convergence.md)).

### Sampling a minibatch

```python
def sample(self, batch_size):
    batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
    states = np.stack([e.state for e in batch])
    policies = np.stack([e.policy for e in batch])
    values = np.array([e.value for e in batch], dtype=np.float32)
    return states, policies, values
```

`random.sample` draws `batch_size` (default 256) examples *without replacement*
and stacks them into three numpy arrays ready for the network:

| Array | Shape | Role |
| --- | --- | --- |
| `states` | `(B, 19, 8, 8)` | network input |
| `policies` | `(B, 4672)` | policy target `π` |
| `values` | `(B,)` | value target `z` |

The `min(batch_size, len(self.buffer))` guard means early on — when the buffer has
fewer than 256 examples — you simply get everything available, so training never
crashes on a nearly-empty buffer.

⚠️ **The buffer is in-memory only.** `ReplayBuffer` *has* `save`/`load` methods…

```python
def save(self, path): ...   # pickle the deque
def load(self, path): ...
```

…but nothing in the training loop calls them. The buffer lives in RAM for the
duration of a run and is discarded when the process exits. In particular, when you
`--resume` a checkpoint ([Chapter 14](14-the-reinforcement-loop.md)), the *network*
is restored but the buffer starts **empty** and refills from fresh self-play. That
is a deliberate simplicity trade-off, and one of the first things you might change
for serious training — see [Chapter 16](16-debugging-and-convergence.md) and
[Chapter 17](17-scaling-and-improvements.md).

## 13.4 The training step, line by line

Here is the whole of `train_epochs`:

```python
def train_epochs(net, buffer, optimizer, config):
    net.train()
    device = config.device

    total_policy_loss = 0.0
    total_value_loss = 0.0
    steps = 0

    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0}

    batches_per_epoch = max(1, len(buffer) // config.batch_size)

    for _ in range(config.epochs_per_iteration):
        for _ in range(batches_per_epoch):
            states, policies, values = buffer.sample(config.batch_size)

            states_t = torch.from_numpy(states).to(device)
            policies_t = torch.from_numpy(policies).to(device)
            values_t = torch.from_numpy(values).to(device)

            policy_logits, value_pred = net(states_t)

            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, values_t)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            steps += 1

    return {
        "policy_loss": total_policy_loss / steps,
        "value_loss": total_value_loss / steps,
    }
```

Let's dissect the important lines.

### `net.train()` — training mode on

The network has `BatchNorm` layers ([Chapter 10](10-the-neural-network.md)).
`net.train()` puts them in training mode (using batch statistics and updating
running averages). Contrast this with self-play and real play, which call
`net.eval()` so BatchNorm uses its stored running statistics and behaves
deterministically. Forgetting this switch is a classic source of "works in
training, weird in play" bugs.

### How many updates? `epochs × batches_per_epoch`

```python
batches_per_epoch = max(1, len(buffer) // config.batch_size)
for _ in range(config.epochs_per_iteration):        # default 4
    for _ in range(batches_per_epoch):
        ...
```

Each *epoch* takes roughly enough minibatches to cover the buffer once
(`len(buffer) // batch_size`), and we do `epochs_per_iteration` (default 4) of
them. So a bigger buffer means more gradient steps per training phase. The
`max(1, ...)` ensures at least one step even when the buffer is tiny.

### Moving data to the device

```python
states_t = torch.from_numpy(states).to(device)
policies_t = torch.from_numpy(policies).to(device)
values_t = torch.from_numpy(values).to(device)
```

`config.device` is `cuda`, `mps`, or `cpu` (chosen by `default_device()` in
[`config.py`](../src/chesszero/config.py)). The numpy batches from the buffer are
converted to tensors and shipped to wherever the network lives.

### The forward pass — two outputs

```python
policy_logits, value_pred = net(states_t)
```

One call, two heads ([Chapter 10](10-the-neural-network.md)):

- `policy_logits` — shape `(B, 4672)`, **raw** scores (not yet probabilities).
- `value_pred` — shape `(B,)`, each a scalar in `[-1, 1]` (the `tanh` value).

### The policy loss — cross-entropy toward the search policy

```python
log_probs = F.log_softmax(policy_logits, dim=1)
policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
```

This is **cross-entropy between the MCTS policy `π` (target) and the network's
predicted move distribution.** Reading it right-to-left:

1. `F.log_softmax(policy_logits)` turns the raw logits into log-probabilities over
   all 4672 actions.
2. `policies_t * log_probs` multiplies each action's target probability `π(a)` by
   the network's log-probability `log p(a)`.
3. `.sum(dim=1)` sums over actions → the (negative) cross-entropy per example.
4. The leading `-` and `.mean()` give the average positive cross-entropy over the
   batch.

📐 Cross-entropy $-\sum_a \pi(a)\log p(a)$ is minimized exactly when the network's
distribution $p$ equals the target $\pi$. So minimizing `policy_loss` teaches the
network to **imitate the search** — to make its cheap, one-shot policy look like
the expensive, look-ahead-refined MCTS policy. This is the *policy-improvement
distillation* at the core of AlphaZero ([Chapter 7](07-the-alphazero-algorithm.md)):
search produces something better than the raw network, and we bake that
improvement back into the weights.

### The value loss — regression toward the outcome

```python
value_loss = F.mse_loss(value_pred, values_t)
```

Plain **mean-squared error** between the value head's prediction and the game's
actual result `z ∈ {-1, 0, +1}`. Minimizing it teaches the network: *from this
position, what outcome should I expect?* This is the only place the true **reward**
enters the weights — everything else is distilled search.

### The combined objective

```python
loss = policy_loss + value_loss
```

A simple sum. Both heads share the residual tower
([Chapter 10](10-the-neural-network.md)), so a single `loss.backward()` sends
gradients from *both* objectives into the shared body — the value signal and the
policy signal jointly shape the network's internal representation of a position.

### The optimizer step and weight decay

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Standard PyTorch: zero old gradients, backpropagate, take a step. The optimizer
itself is created in the training loop ([Chapter 14](14-the-reinforcement-loop.md)):

```python
torch.optim.Adam(net.parameters(),
                 lr=config.learning_rate,       # 1e-3
                 weight_decay=config.weight_decay)  # 1e-4
```

That `weight_decay` is **L2 regularization** — it gently pulls all weights toward
zero each step, discouraging the network from overfitting the (noisy) self-play
labels. It is applied invisibly by Adam, which is why you never see it in the loss
expression.

## 13.5 Mapping the code to the AlphaZero loss

[Chapter 7](07-the-alphazero-algorithm.md) wrote the AlphaZero objective as:

$$
L = \underbrace{(z - v)^2}_{\text{value}} \;\;\underbrace{-\; \boldsymbol{\pi}^\top \log \mathbf{p}}_{\text{policy}} \;\;+\; \underbrace{c\lVert\theta\rVert^2}_{\text{regularization}}
$$

Every term is present in the code, spread across two places:

| Loss term | Code | Where |
| --- | --- | --- |
| $(z - v)^2$ | `F.mse_loss(value_pred, values_t)` | `train.py` |
| $-\boldsymbol{\pi}^\top \log \mathbf{p}$ | `-(policies_t * log_probs).sum(dim=1).mean()` | `train.py` |
| $c\lVert\theta\rVert^2$ | `weight_decay=config.weight_decay` in the Adam optimizer | `cli.py` |

So the humble `loss = policy_loss + value_loss`, plus Adam's weight decay, *is* the
AlphaZero loss. Theory and implementation line up exactly.

## 13.6 Two heads, two kinds of signal

It is worth stating plainly, because it is the conceptual crux of the whole
method:

```
                 SELF-PLAY  (Chapter 12)
                         │
        ┌────────────────┴─────────────────┐
        ▼                                   ▼
   π  = MCTS visit counts             z = game outcome ∈ {-1,0,1}
   (a DISTILLATION of search)         (the true REWARD, ground truth)
        │                                   │
        ▼                                   ▼
   policy head  ◄── cross-entropy      value head ◄── MSE
   "imitate the search"                "predict the result"
```

- The **value head** learns from the genuine reward `z`. This is the RL signal —
  the only thing anchored to reality (who actually won).
- The **policy head** learns from `π`, which is *not* a reward but a *better
  policy* produced by search. This is supervised imitation of a teacher that is a
  stronger version of the student itself.

Both improve together, iteration after iteration, which — as
[Chapter 5](05-self-play-and-games.md) and [Chapter 7](07-the-alphazero-algorithm.md)
argued — is why the whole loop climbs from random play toward strong play.

## 13.7 A real subtlety: cross-entropy over illegal moves

Look again at the policy loss:

```python
log_probs = F.log_softmax(policy_logits, dim=1)   # softmax over ALL 4672 actions
policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
```

⚠️ The `log_softmax` normalizes over the **entire 4672-action space**, including
moves that are *illegal* in the given position. Contrast this with play time,
where the `Evaluator` masks to legal moves *before* softmaxing
([Chapter 9](09-encoding-board-and-moves.md), [Chapter 11](11-mcts-in-code.md)).

Is that a bug? No — and it is worth understanding why:

- The **target** `π` from self-play is zero on every illegal move (MCTS only ever
  visits legal moves). So the term `π(a)·log p(a)` contributes exactly `0` for
  every illegal `a`; illegal moves never directly push the loss up or down.
- But illegal-move logits *do* sit in the softmax denominator, so the network is
  implicitly encouraged to keep their probability low (any mass there is mass
  stolen from the legal moves it is being trained to match).

The net effect: over training, the network learns to assign near-zero probability
to illegal moves on its own — a nice emergent behaviour — even though we never
mask during training. And it does not matter for *correctness of play* regardless,
because at play time we *always* mask to legal moves before choosing. Training
over the full action space is simpler (no per-example legal mask needs to be
stored or applied) and is exactly what the original AlphaZero does.

## 13.8 What `train_epochs` returns

```python
return {
    "policy_loss": total_policy_loss / steps,
    "value_loss": total_value_loss / steps,
}
```

The mean policy and value losses over all the gradient steps in this training
phase. These two numbers are printed by the loop each iteration and are your
primary window into whether learning is healthy — reading them is the subject of
[Chapter 16](16-debugging-and-convergence.md).

Next we zoom out one level and see how self-play (Phase A) and this training step
(Phase B) are stitched into the full reinforcement loop, iteration after
iteration, with checkpoints saved along the way.

---

## Key takeaways

- Each self-play position is an **`Example(state, policy, value)`** — network
  input, policy target `π`, and value target `z`.
- The **replay buffer** is a fixed-size sliding window (default 50k) that
  decorrelates samples, reuses expensive data, and fights forgetting. It is
  **in-memory only** and not restored on resume.
- `train_epochs` samples minibatches and minimizes
  **`policy_loss + value_loss`** = cross-entropy(logits, `π`) + MSE(value, `z`),
  with Adam applying L2 **weight decay** — exactly the AlphaZero loss.
- The **value head learns from the true reward**; the **policy head imitates the
  search policy**. Two heads, two kinds of signal, one shared body.
- The policy cross-entropy spans all 4672 actions, but zero-valued illegal targets
  make this harmless — and play always masks to legal moves anyway.

## Exercises

1. With `batch_size = 256`, `epochs_per_iteration = 4`, and a buffer holding
   3,000 examples, roughly how many gradient steps does one `train_epochs` call
   take?
2. Why does `train_epochs` call `net.train()` while self-play and real play call
   `net.eval()`? What component's behaviour changes, and what would go wrong if you
   used the wrong mode?
3. The value loss uses MSE against `z ∈ {-1, 0, 1}` while the value head outputs a
   continuous `tanh` in `[-1, 1]`. What does an intermediate prediction like `0.3`
   *mean*, and is it a problem that `z` is never actually `0.3`?
4. Suppose you removed the replay buffer and trained only on the single most recent
   game each iteration. Name two things from §13.3 that would break.
5. Explain, using §13.7, why the network gradually learns to give illegal moves
   near-zero probability even though we never mask logits during training.

---

> **Course:** [Home](README.md) · **Prev:** [12. Self-Play](12-self-play-in-code.md) · **Next:** [14. The Reinforcement Loop & Checkpoints](14-the-reinforcement-loop.md)
