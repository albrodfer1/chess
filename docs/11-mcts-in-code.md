# Chapter 11 — Monte Carlo Tree Search in Code

> **Course:** [Home](README.md) · **Prev:** [10. The Neural Network](10-the-neural-network.md) · **Next:** [12. Self-Play](12-self-play-in-code.md)

**What you'll learn**
- How the `Evaluator` turns raw network logits into a **legal-masked** policy
- The `Node` data structure — and why an unvisited node's value is exactly `0`
- The four phases of `run_mcts`, line by line: expand, select, evaluate, back up
- Why PUCT scores a child as `-child.value`, and why back-ups flip the sign
- How visit counts become an improved **search policy**, and how a move is chosen

---

## 11.1 Where we are

[Chapter 6](06-monte-carlo-tree-search.md) gave you the theory of Monte Carlo
Tree Search: a budget of simulations, PUCT selection, a neural value estimate at
the leaves instead of random rollouts, and visit counts as an improved policy.
This chapter is that theory made real, in [`mcts.py`](../src/chesszero/mcts.py).

This is the module that fulfills **requirement 3**: a position's *value* comes
from Monte Carlo Tree Search. The network offers a fast opinion; MCTS turns it
into a considered judgment by looking ahead. It also quietly upholds
**requirement 1** — only legal moves ever enter the tree.

Keep the file open beside this chapter. We'll walk every function.

## 11.2 The `Evaluator`: the network's opinion, masked to legality

Before MCTS can grow a tree it needs, at any position, two things from the
network: a **prior** over the legal moves and a **value** for the position. The
`Evaluator` wraps [`ChessNet`](10-the-neural-network.md) to provide exactly that.

```python
class Evaluator:
    def __init__(self, net: ChessNet, device: str) -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """Return ({legal_move: prior_prob}, value_for_side_to_move)."""
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        logits = logits[0].cpu().numpy()

        moves = list(board.legal_moves)
        if not moves:
            return {}, float(value.item())

        indices = np.fromiter((move_to_index(m) for m in moves), dtype=np.int64)
        move_logits = logits[indices]
        # Softmax over legal moves only -> illegal moves get zero probability.
        move_logits -= move_logits.max()
        priors = np.exp(move_logits)
        priors /= priors.sum()

        return {m: float(p) for m, p in zip(moves, priors)}, float(value.item())
```

Read it top to bottom:

1. **Encode** the board to a `(19, 8, 8)` tensor ([Chapter 9](09-encoding-board-and-moves.md)),
   add a batch dimension with `unsqueeze(0)`, and move it to the device.
2. **Run the network** ([Chapter 10](10-the-neural-network.md)). Because we're only
   *reading* the network here (no training), the whole method is wrapped in
   `@torch.no_grad()` — this skips building the autograd graph, saving time and
   memory.
3. **List the legal moves** with `board.legal_moves`. `python-chess` gives us
   exactly the legal set — the source of our legality guarantee.
4. **Mask** — this is the crucial step. We compute `move_to_index` for each legal
   move and pull out *only those* logits from the 4672-vector. Every illegal move's
   logit is simply never looked at.
5. **Softmax over the legal logits only.** Subtracting `move_logits.max()` is the
   standard trick to avoid `exp()` overflow; it doesn't change the result. After
   normalization the priors sum to 1 over the legal moves, and illegal moves have
   an implicit probability of exactly **zero**.

So the `Evaluator` returns a dictionary `{legal_move: prior}` and a single
`value` (the side-to-move scalar from [Chapter 10](10-the-neural-network.md)).

> ⚠️ This masked-softmax is *the* place requirement 1 is enforced during search.
> The raw policy head knows nothing of legality; the `Evaluator` guarantees MCTS
> only ever grows legal branches.

## 11.3 The `Node`: one position in the tree

Each node of the search tree stores the statistics MCTS accumulates about one
position.

```python
class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "parent", "move", "board")

    def __init__(self, prior, parent=None, move=None):
        self.prior = prior            # P(this move) from the network, set by the parent
        self.visit_count = 0          # N: how many simulations passed through
        self.value_sum = 0.0          # W: sum of backed-up values
        self.children = {}            # move -> Node
        self.parent = parent
        self.move = move              # the move that leads here from the parent
        self.board = None             # filled in lazily (see §11.5)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

A few design notes:

- **`__slots__`** avoids a per-node `__dict__`, shrinking memory. MCTS creates a
  *lot* of nodes, so this matters.
- **`value` is a running average** — `value_sum / visit_count`. This is the "Monte
  Carlo" part: a node's value is the mean of every leaf value ever backed up
  through it (see [Chapter 6](06-monte-carlo-tree-search.md)).
- **An unvisited node has `value == 0`.** With `visit_count == 0` the property
  returns `0.0` rather than dividing by zero. This zero-initialization has a
  beautiful consequence, explained next.

📐 **Why the zero start matters.** In PUCT (§11.6) a child is scored as
`Q + U`, where `Q = -child.value` and `U` grows with the child's prior. For a
never-visited child, `Q = 0`, so its whole score is the exploration term `U`,
which is proportional to its **prior**. In other words, *before it has any
evidence of its own, a move is judged purely by what the network thinks of it.*
Search naturally starts from the network's intuition and refines it.

## 11.4 Terminal positions: real results, not estimates

When a simulation reaches a finished game, we don't need the network's guess — we
know the truth.

```python
def terminal_value(board: chess.Board) -> float:
    """Game result from the perspective of the side to move at `board`."""
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    # If it's a win, the side to move can't be the winner (they were mated).
    return 1.0 if outcome.winner == board.turn else -1.0
```

The subtlety is the perspective. `board.outcome().winner` is the color that won.
At a checkmate the side *to move* is the one who got mated, so if `winner == turn`
that would be impossible for a checkmate — the value is `-1.0` from the mover's
view (they're the loser). Draws (`winner is None`) return `0.0`. As always in this
codebase, value is **from the perspective of the player to move**, matching the
network's value head ([Chapter 10](10-the-neural-network.md)).

## 11.5 The heart: `run_mcts`

Everything comes together in `run_mcts`. Given a board, an evaluator, and the
config, it grows a tree of `num_simulations` simulations and returns the root.

```python
def run_mcts(board, evaluator, config, add_noise=False) -> Node:
    root = Node(prior=0.0)
    root.board = board.copy()

    priors, _ = evaluator.evaluate(root.board)
    _expand(root, priors)
    if add_noise and root.children:
        _add_dirichlet_noise(root, config)

    for _ in range(config.num_simulations):
        node = root
        search_path = [root]

        # Selection: descend until we reach an unexpanded node.
        while node.expanded:
            node = _select_child(node, config.c_puct)
            search_path.append(node)

        # Lazily materialize the board for the reached leaf.
        if node.board is None:
            node.board = node.parent.board.copy()
            node.board.push(node.move)

        if node.board.is_game_over():
            value = terminal_value(node.board)
        else:
            priors, value = evaluator.evaluate(node.board)
            _expand(node, priors)

        # Backup: alternate sign each ply back up the tree.
        for path_node in reversed(search_path):
            path_node.visit_count += 1
            path_node.value_sum += value
            value = -value

    return root
```

Let's take it in pieces — these are the four phases of MCTS from
[Chapter 6](06-monte-carlo-tree-search.md).

### Setup: create and expand the root

```python
root = Node(prior=0.0)
root.board = board.copy()
priors, _ = evaluator.evaluate(root.board)
_expand(root, priors)
```

We copy the board (MCTS must never mutate the caller's board), ask the evaluator
for the priors, and **expand** the root — creating a child `Node` for every legal
move, each seeded with its prior:

```python
def _expand(node: Node, priors: dict[chess.Move, float]) -> None:
    for move, prob in priors.items():
        node.children[move] = Node(prior=prob, parent=node, move=move)
```

Since `priors` only contains legal moves (§11.2), the tree can only ever branch on
legal moves — requirement 1 again.

### Root exploration noise (self-play only)

```python
if add_noise and root.children:
    _add_dirichlet_noise(root, config)
```

```python
def _add_dirichlet_noise(root: Node, config: Config) -> None:
    moves = list(root.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(moves))
    eps = config.dirichlet_epsilon
    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = (1 - eps) * child.prior + eps * float(n)
```

This blends a little randomness into the root priors so self-play games explore
different openings and don't collapse into the same line every time
([Chapter 5](05-self-play-and-games.md)). It is applied **only at the root**, and
**only during self-play** (`add_noise=True`). When the agent plays for real
([Chapter 15](15-playing-evaluating-viewer.md)) `add_noise` is `False` — you want
its honest best move, not deliberate exploration.

### Phase 1 — Selection

```python
node = root
search_path = [root]
while node.expanded:
    node = _select_child(node, config.c_puct)
    search_path.append(node)
```

Starting at the root, we repeatedly pick the "best" child by PUCT until we fall off
the bottom of the tree onto an **unexpanded** node (one with no children yet). We
remember the whole route in `search_path` so we can back up along it later.

### Phase 2 — Expansion & evaluation

```python
if node.board is None:
    node.board = node.parent.board.copy()
    node.board.push(node.move)

if node.board.is_game_over():
    value = terminal_value(node.board)
else:
    priors, value = evaluator.evaluate(node.board)
    _expand(node, priors)
```

First we **lazily materialize** the leaf's board. Child nodes are created without a
board (to save the cost of copying a board for branches we might never visit); the
first time we actually reach a node, we build its board by copying the parent's and
pushing the move. The parent always has a board because it was expanded earlier.

Then the leaf is scored:

- If the game is **over**, use the exact `terminal_value` (§11.4).
- Otherwise, ask the network for `(priors, value)` and **expand** the leaf with its
  children. Note we grabbed the network's `value` here — *this is the bootstrap
  from [Chapter 6](06-monte-carlo-tree-search.md)*: instead of playing the game out
  randomly, we trust the value head's estimate. This is why even a handful of
  simulations still yields a value.

### Phase 3 — Back-up

```python
for path_node in reversed(search_path):
    path_node.visit_count += 1
    path_node.value_sum += value
    value = -value
```

We walk from the leaf back to the root. At each node we record one more visit and
add the leaf value to its `value_sum`. The magical line is `value = -value`: the
sign flips at every ply.

📐 **Why flip?** Value is always *from the mover's perspective*. If the leaf is
good for the player to move there (`value = +0.7`), it is exactly that bad for the
opponent one ply up (`-0.7`), good again for us two plies up, and so on. Flipping
the sign keeps each ancestor's `value_sum` in *its own* perspective. This is the
zero-sum reasoning from [Chapter 5](05-self-play-and-games.md), expressed in one
line.

## 11.6 PUCT selection: `_select_child`

```python
def _select_child(node: Node, c_puct: float) -> Node:
    best_score = -float("inf")
    best_child = None
    sqrt_total = math.sqrt(node.visit_count)
    for child in node.children.values():
        q = -child.value
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child
```

Each child is scored as `score = Q + U`:

$$\text{score}(a) = \underbrace{-\,\text{child.value}}_{Q:\ \text{exploit}} \;+\; \underbrace{c_{\text{puct}}\cdot P(a)\cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(a)}}_{U:\ \text{explore}}$$

- **`q = -child.value`** — the exploitation term. We *negate* because the child's
  value is stored from the child mover's perspective (the opponent's), so from the
  parent's point of view a move is good exactly when it's bad for the opponent.
  This negation and the back-up's sign flip are the same idea seen from two angles.
- **`u = c_puct * prior * sqrt(N_parent) / (1 + N_child)`** — the exploration term.
  It is large when a move has a high **prior** but few visits, and it shrinks as the
  child is visited more. `c_puct` (default `1.5` in
  [`config.py`](../src/chesszero/config.py)) tunes how much we explore versus
  exploit.

As noted in §11.3, an unvisited child has `Q = 0`, so early on selection follows
the network's prior; as visits accumulate, `Q` takes over and the search
concentrates on lines that actually evaluate well.

## 11.7 From a tree to a decision

After `run_mcts` returns, the root's children carry **visit counts** — and the
visit distribution *is* the improved policy MCTS produced
([Chapter 6](06-monte-carlo-tree-search.md)). Two helpers turn it into something
usable.

### `policy_from_visits` — the training target π

```python
def policy_from_visits(root: Node, temperature: float = 1.0) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if visits.sum() == 0:
        return policy

    if temperature <= 1e-6:
        best = moves[int(visits.argmax())]
        policy[move_to_index(best)] = 1.0
        return policy

    scaled = visits ** (1.0 / temperature)
    scaled /= scaled.sum()
    for move, p in zip(moves, scaled):
        policy[move_to_index(move)] = p
    return policy
```

This produces the `4672`-vector `π` recorded during self-play and later used to
train the policy head ([Chapters 12](12-self-play-in-code.md) and
[13](13-training-and-replay-buffer.md)). The **temperature** reshapes the
distribution:

- `temperature = 1` → proportional to visit counts (the honest search policy).
- `temperature → 0` → a one-hot vector on the single most-visited move (fully
  greedy). The `visits ** (1/temperature)` sharpens toward the top move as
  temperature drops.

### `select_move` — actually pick a move to play

```python
def select_move(root: Node, temperature: float = 0.0) -> chess.Move:
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if temperature <= 1e-6:
        return moves[int(visits.argmax())]           # greedy: most-visited move

    probs = visits ** (1.0 / temperature)
    probs /= probs.sum()
    return moves[int(np.random.choice(len(moves), p=probs))]  # sampled
```

With `temperature = 0` (used for real play and late in self-play games) it returns
the **most-visited** move — MCTS's best answer. With `temperature = 1` (used for the
opening plies of self-play) it **samples** proportional to visits, injecting
variety into training games. Either way, the returned move is one of the root's
children, and every child is a legal move — so the agent still can never play an
illegal move.

## 11.8 The big picture: this is where value comes from

Step back and see what just happened. For a single move decision, `run_mcts`:

```
        current position
              │  100 simulations, each:
              │    select ─► reach a leaf ─► evaluate (net value OR terminal)
              │            ─► expand ─► back up (averaging, sign-flipping)
              ▼
   root with visit counts + a refined value on every node
              │
   policy_from_visits ──► π   (improved policy, → training)
   select_move        ──► the move to play
```

- The **network's value** at the leaves is the raw material; the **search**
  averages it over many imagined lines into a much better estimate (requirement 3).
- The **depth is adaptive**: there is no fixed lookahead horizon. The only budget is
  `num_simulations`, and PUCT decides where to spend it — deep on forcing lines,
  shallow elsewhere. More simulations means a stronger move (and a slower one).
- The **opponent is modeled inside the tree**: the sign-flipping back-up means every
  other ply is scored from the opponent's perspective, so search plans against an
  opponent playing just as well — no external adversary needed
  ([Chapter 5](05-self-play-and-games.md)).

With this, our agent can think. Next we use it to play whole games against itself
and turn those games into training data.

---

## Key takeaways

- The `Evaluator` runs the network and **softmaxes over legal moves only**, so MCTS
  can never branch on an illegal move (requirement 1).
- A `Node` tracks `prior`, `visit_count`, and `value_sum`; its `value` is the
  running average of backed-up leaf values, and is **0 when unvisited** so search
  starts from the network's prior.
- `run_mcts` repeats four phases — select (PUCT), expand, evaluate (network value or
  exact terminal), back up — with a **sign flip** each ply to respect the zero-sum,
  side-to-move value convention.
- PUCT scores a child as `-child.value + c_puct·prior·√N_parent/(1+N_child)`,
  balancing exploitation against prior-weighted exploration.
- **Visit counts** become the improved policy `π` (`policy_from_visits`) and choose
  the move (`select_move`); this refined value/policy is the payoff of requirement 3.

## Exercises

1. Trace the value of a leaf worth `+0.4` (from that leaf's mover) as it backs up
   four plies to the root. What value does each ancestor receive?
2. Why is `run_mcts` careful to `board.copy()` before pushing moves? What bug would
   appear if it pushed onto the caller's board instead?
3. A child has `prior = 0.6` but has never been visited. Another has `prior = 0.05`
   and `value = 0.9` after 10 visits. Write out both PUCT scores (assume
   `c_puct = 1.5`, `N_parent = 20`) and say which gets selected.
4. In `select_move`, what exactly changes between `temperature = 0` and
   `temperature = 1`? Why does self-play want `1` early and `0` late?
5. Set `num_simulations = 1` in `config.py`. What does the resulting move policy
   essentially reduce to? (Hint: how much better than the raw network prior can one
   simulation make it?)

---

> **Course:** [Home](README.md) · **Prev:** [10. The Neural Network](10-the-neural-network.md) · **Next:** [12. Self-Play](12-self-play-in-code.md)
