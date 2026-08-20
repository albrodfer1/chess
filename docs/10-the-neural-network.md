# Chapter 10 — The Neural Network

> **Course:** [Home](README.md) · **Prev:** [9. Encoding the Board & Moves](09-encoding-board-and-moves.md) · **Next:** [11. Monte Carlo Tree Search in Code](11-mcts-in-code.md)

**What you'll learn**
- What `ChessNet` computes: one board in, a policy *and* a value out
- Why we use **convolutions** on the 8×8 board, and what a **residual block** buys us
- How the shared "tower" splits into a **policy head** and a **value head**
- The exact tensor shapes flowing through `forward()`, step by step
- Which knobs (`num_res_blocks`, `num_filters`) trade playing strength for speed

---

## 10.1 The network's job

In [Chapter 4](04-deep-rl-and-function-approximation.md) we argued that chess has
far too many positions to store a value for each one, so we must *approximate* a
function over positions with a neural network. In [Chapter 7](07-the-alphazero-algorithm.md)
we saw that AlphaZero uses a single network with **two outputs**. This chapter is
that network in code: [`network.py`](../src/chesszero/network.py).

Formally, the network is a function

$$f_\theta(s) = (\mathbf{p},\, v)$$

- `s` is an encoded board — the `(19, 8, 8)` tensor from
  [Chapter 9](09-encoding-board-and-moves.md).
- `p` (the **policy**) is a vector of `4672` numbers, one per possible move in the
  action space. High numbers mean "this move looks promising."
- `v` (the **value**) is a single number in `[-1, 1]` estimating how good the
  position is *for the player to move* — `+1` "I am winning," `-1` "I am losing,"
  `0` "even."

θ (theta) is the set of learnable weights. Training (Chapter 13) adjusts θ so that
`p` matches the moves MCTS preferred and `v` matches who actually won.

```
                                          ┌─────────────► policy p  (4672 logits)
   board ──► encode ──► (19,8,8) ──► [ ChessNet body ] ─┤
                                          └─────────────► value v   (1 scalar, tanh)
```

Two outputs from one shared body: this is the classic **two-headed** (multi-task)
design. The body learns a general "understanding" of the position once, and both
heads read from it.

## 10.2 Why convolutions?

A chess board is an 8×8 grid, and the useful patterns in chess are **local and
translation-similar**: a knight fork looks like a knight fork whether it happens
on the kingside or the queenside; a pawn chain is a pawn chain anywhere on the
board. This is exactly the situation convolutional neural networks (CNNs) were
designed for.

A convolution slides a small filter (here 3×3) across the board and computes a
response at every square. Two properties make this ideal:

- **Locality** — each output cell depends only on a small neighborhood, matching
  how chess tactics are local interactions between nearby pieces.
- **Weight sharing** — the *same* filter is applied at every square, so a pattern
  learned in one place is instantly recognized everywhere. This is enormously more
  efficient than a fully-connected layer that would have to relearn the pattern
  for each of the 64 squares independently.

Because our board tensor keeps its spatial `8×8` shape across all 19 planes (see
[Chapter 9](09-encoding-board-and-moves.md)), a CNN can look at "which pieces are
next to which" directly.

## 10.3 The residual block

Deep networks are more expressive, but stacking many plain convolution layers
makes them hard to train — gradients shrink or explode on their way back, and
accuracy can actually get *worse* with depth. The fix, from the famous ResNet
architecture, is the **residual block**: instead of asking each block to compute a
whole new representation, we ask it to compute a *change* to add to its input.

Here is the real code:

```python
class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)          # <-- the "skip" connection
```

Three things are happening here, and each earns its place:

- **`Conv2d(filters, filters, 3, padding=1)`** — a 3×3 convolution. `padding=1`
  keeps the board at 8×8 (it doesn't shrink), so every block preserves spatial
  size. `bias=False` because the following BatchNorm has its own shift term, so a
  bias would be redundant.
- **`BatchNorm2d`** — batch normalization. It re-centers and re-scales each
  channel across the batch, which keeps activations in a healthy range and makes
  training much faster and more stable.
- **`x + residual`** — the **skip connection**. The block's convolutions compute a
  *correction*, and we add it back to the untouched input. If the best thing a
  block can do is nothing, it can easily learn to output ≈0 and pass its input
  through unchanged. This is what lets us stack many blocks safely.

📐 *Why the skip helps the gradient.* During backpropagation, the `+ residual`
term gives the gradient a direct "highway" back to earlier layers — the derivative
of `x + f(x)` with respect to `x` always includes a `1`, so the signal never fully
vanishes no matter how deep the tower is.

> 📖 **Deep Dive:** For a full mathematical derivation of the gradient highway, receptive field growth on the 8×8 board, and why ResNets outperform MLPs/Transformers here, see [Extra: Residual Blocks Deep Dive](Extra/residual-blocks.md).

The activation function throughout is **ReLU** (`max(0, x)`), a cheap
nonlinearity that lets the network represent complex functions.

## 10.4 The whole network, `ChessNet`

Now the full model. It has three parts: a **stem** that lifts the 19 input planes
up to `num_filters` channels, a **tower** of residual blocks, and two **heads**.

```python
class ChessNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        f = config.num_filters

        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, f, 3, padding=1, bias=False),  # 19 -> f
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(f) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(f, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 64, config.action_size)   # -> 4672

        # Value head
        self.value_conv = nn.Conv2d(f, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)                       # -> 1
```

Here is the shape of data as it flows through, using the default
`num_filters = 128` from [`config.py`](../src/chesszero/config.py):

```
input                 (B, 19, 8, 8)     B = batch size
  │  stem: Conv 19->128, BN, ReLU
  ▼
                      (B, 128, 8, 8)
  │  num_res_blocks × ResidualBlock  (shape unchanged)
  ▼
   shared features    (B, 128, 8, 8)
  ├──────────────────────────────┐
  ▼ policy head                   ▼ value head
Conv 128->2, BN, ReLU           Conv 128->1, BN, ReLU
  (B, 2, 8, 8)                    (B, 1, 8, 8)
flatten -> (B, 128)             flatten -> (B, 64)
Linear 128 -> 4672              Linear 64 -> 256, ReLU
  (B, 4672)  policy_logits      Linear 256 -> 1, tanh
                                  (B,)       value
```

The two `1×1` convolutions in the heads are a standard trick to *compress* the
128-channel feature map down to a couple of channels before flattening — this
keeps the fully-connected layers small.

## 10.5 The policy head

```python
p = F.relu(self.policy_bn(self.policy_conv(x)))   # (B, 2, 8, 8)
p = p.reshape(p.size(0), -1)                       # (B, 128)
policy_logits = self.policy_fc(p)                  # (B, 4672)
```

The output is a vector of `4672` **logits** — one raw score per action in the
encoding from [Chapter 9](09-encoding-board-and-moves.md). Note what it is **not**:

- ⚠️ **These are not probabilities.** They are unbounded real numbers. You cannot
  read them as "move probabilities" yet.
- ⚠️ **They are over *all* moves, legal or not.** The network has no built-in
  concept of legality; it happily produces a score for moving a knight off the
  board.

Both problems are fixed *outside* the network, in the `Evaluator` you'll meet in
[Chapter 11](11-mcts-in-code.md): it selects only the logits at **legal** move
indices, subtracts the max for numerical stability, exponentiates, and normalizes.
That masked softmax is what guarantees the agent can never even *consider* an
illegal move (requirement 1, [Chapter 9](09-encoding-board-and-moves.md)). Keeping
legality out of the network keeps the network simple and the guarantee airtight.

## 10.6 The value head

```python
v = F.relu(self.value_bn(self.value_conv(x)))   # (B, 1, 8, 8)
v = v.reshape(v.size(0), -1)                      # (B, 64)
v = F.relu(self.value_fc1(v))                     # (B, 256)
value = torch.tanh(self.value_fc2(v)).squeeze(-1) # (B,)
```

The value head funnels the shared features down to a *single number* per position
and squashes it through **`tanh`**, which bounds the output to `[-1, 1]`. That
range is chosen to match the reward: `+1` win, `-1` loss, `0` draw.

Crucially, the value is **from the perspective of the side to move**. If it's
Black's turn and `value ≈ +0.8`, that means "Black is doing well here," not "White."
This side-to-move convention is exactly what makes the **sign flips** in MCTS work
(the `q = -child.value` and `value = -value` you'll see in
[Chapter 11](11-mcts-in-code.md)), and it ties back to the zero-sum reasoning in
[Chapter 5](05-self-play-and-games.md).

⚠️ The value is a *learned estimate*, not the truth. Early in training it is
essentially noise. Only the final **game outcome** `z` is ground truth — and it's
`z` that teaches the value head during training ([Chapter 13](13-training-and-replay-buffer.md)).

## 10.7 Putting it together: `forward()`

The whole forward pass is short enough to read in one glance:

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = self.stem(x)
    for block in self.res_blocks:
        x = block(x)

    p = F.relu(self.policy_bn(self.policy_conv(x)))
    p = p.reshape(p.size(0), -1)
    policy_logits = self.policy_fc(p)

    v = F.relu(self.value_bn(self.value_conv(x)))
    v = v.reshape(v.size(0), -1)
    v = F.relu(self.value_fc1(v))
    value = torch.tanh(self.value_fc2(v)).squeeze(-1)

    return policy_logits, value
```

Notice the shared body (`stem` + `res_blocks`) runs **once**, and both heads read
its output `x`. One forward pass yields both the policy and the value — that's the
efficiency of the two-headed design in action.

## 10.8 The knobs: capacity vs. speed

Two fields in [`config.py`](../src/chesszero/config.py) set the size of the tower:

```python
num_res_blocks: int = 6      # how DEEP the tower is
num_filters: int = 128       # how WIDE each layer is (channels)
```

| Knob | Bigger means… | Cost |
| --- | --- | --- |
| `num_res_blocks` | more layers → can represent deeper chess patterns | slower forward pass, harder to train |
| `num_filters` | more channels → more patterns per layer | quadratically more compute and memory |

The real AlphaZero used ~20 blocks and 256 filters. Our defaults (6 blocks, 128
filters) are deliberately modest so the whole loop runs on a laptop. This is the
single biggest lever on playing strength versus wall-clock time — we return to
scaling it in [Chapter 17](17-scaling-and-improvements.md).

Remember, too, that the network is called *constantly*: MCTS invokes it once per
tree leaf, and it runs `num_simulations` (default 100) times **per move**
([Chapter 11](11-mcts-in-code.md)). A network twice as slow makes the entire
self-play phase roughly twice as slow. Size it accordingly.

---

## Key takeaways

- `ChessNet` maps one encoded board `(19, 8, 8)` to a **policy** (`4672` logits)
  and a **value** (one scalar in `[-1, 1]`) — two heads on a shared body.
- **Convolutions** exploit the board's spatial, translation-similar structure;
  **residual blocks** (skip connections + BatchNorm) let us stack many of them and
  still train.
- The policy head emits raw **logits over all moves** — masking to legal moves and
  softmax happen later, in the `Evaluator` ([Chapter 11](11-mcts-in-code.md)).
- The value head uses **`tanh`** to output a side-to-move evaluation in `[-1, 1]`;
  it is a learned estimate, grounded only later by real game outcomes.
- `num_res_blocks` (depth) and `num_filters` (width) trade playing strength for
  speed and are the main scaling knobs ([Chapter 17](17-scaling-and-improvements.md)).

## Exercises

1. With `num_filters = 128`, what is the shape of the tensor just after the stem?
   And after the policy head's `1×1` conv, before flattening? (Check against §10.4.)
2. The policy head flattens `(B, 2, 8, 8)` to `(B, 128)` before the linear layer to
   4672. Why `128`? Where does that number come from?
3. Suppose you removed the `tanh` from the value head. What could go wrong when the
   value is later compared against a target in `{-1, 0, +1}`? (Hint: what range can
   a bare `Linear` output take?)
4. The network never sees the concept of "legal move." Argue why that's a *feature*
   of the design, not a bug. Where is legality actually enforced?
5. Change `num_res_blocks` to `2` and `num_filters` to `32` in `config.py` and run
   `poetry run pytest`. Do the tests still pass? What would you expect to happen to
   playing strength and to speed?

---

> **Course:** [Home](README.md) · **Prev:** [9. Encoding the Board & Moves](09-encoding-board-and-moves.md) · **Next:** [11. Monte Carlo Tree Search in Code](11-mcts-in-code.md)
