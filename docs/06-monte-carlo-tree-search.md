# Chapter 6 — Monte Carlo Tree Search

> **Course:** [Home](README.md) · **Prev:** [5. Self-Play & Games](05-self-play-and-games.md) · **Next:** [7. The AlphaZero Algorithm](07-the-alphazero-algorithm.md)

**What you'll learn**
- Why we cannot just search the whole game tree, and what to do instead
- The four phases of classic Monte Carlo Tree Search (MCTS)
- The exploration/exploitation dilemma and the **UCT** formula that resolves it
- AlphaZero's key upgrade: replacing random rollouts with a **value network**
- The **PUCT** rule that lets a policy network *guide* the search
- How visit counts become an *improved policy*, and why values are averaged and sign-flipped

---

## 6.1 The problem: the tree is too big to walk

In [Chapter 5](05-self-play-and-games.md) we met the **game tree**: the root is
the current position, each edge is a move, and each node is the position that
results. Minimax says: to find the best move, explore the whole tree, assume the
opponent plays optimally, and back the values up.

There is just one problem. Chess has a **branching factor** of about 35 (roughly
35 legal moves in a typical position), and games run 80+ plies. The number of
leaves in a full-depth tree is around $35^{80}$ — a number with more digits than
this chapter has words. You cannot walk it. You cannot walk a millionth of a
millionth of it.

So we give up on *completeness* and settle for *focus*. Instead of "search
everything," we spend a fixed **budget** of effort — say 100 or 800 "thinking
steps" — and we spend it *where it matters*: on the moves that look promising,
searched deeply, while barely glancing at moves that look bad.

That is exactly what **Monte Carlo Tree Search (MCTS)** does. It grows a small,
lopsided tree — deep along good lines, shallow everywhere else — one simulation
at a time.

```
        Full minimax tree              MCTS tree after 100 simulations
        (impossible to store)          (grown where it matters)

              (root)                            (root)
           ╱ ╱ │ ╲ ╲ ...                     ╱   │    ╲
          ╱ ╱  │  ╲ ╲                       ╱    │     ╲
   every move, every                   good   ok      bad
   reply, forever...                   move   move    move
                                       ╱│╲     │        ·   ← visited once, ignored
                                      ╱ │ ╲    ·
                                    deep, detailed
```

> **The word "budget."** In this project the budget is
> [`config.num_simulations`](../src/chesszero/config.py) (default **100**). Each
> simulation adds roughly one node to the tree. More budget → a bigger, deeper,
> stronger search. We'll return to this in §6.9.

## 6.2 One simulation, four phases

MCTS builds its tree by repeating a single routine — a **simulation** — over and
over. Each simulation walks from the root down to a leaf, adds a new node, and
carries a value back up. Classic MCTS has four phases:

```
   1. SELECTION            2. EXPANSION         3. SIMULATION        4. BACKPROP
   walk down the tree      add a new child      estimate the         push the value
   picking the "best"      node for an          leaf's value         back up the path,
   child at each step      unexplored move                           updating each node
   until you reach a
   not-yet-expanded node

     (root)                  (root)               (root)               (root) ← +v
       │                       │                    │                    │
       ▼                       ▼                    ▼                    ▼ ← −v
     node                    node                 node                 node
       │                       │                    │                    │
       ▼                       ▼                    ▼                    ▼ ← +v
    ✿ leaf                  ✿ leaf               ✿ leaf               ✿ leaf
                              │                    │
                              ▼                    ▼
                           new child            estimate v
```

1. **Selection.** Starting at the root, repeatedly pick a child move according to
   a selection rule (§6.3) until you reach a node that has not been expanded yet.
2. **Expansion.** Create children for that node — one per legal move.
3. **Simulation (evaluation).** Estimate how good the reached position is. In
   *classic* MCTS this meant playing random moves to the end of the game (a
   "rollout") and using the result. AlphaZero replaces this — see §6.5.
4. **Backpropagation.** Send the estimated value back up the path you came down,
   updating every node's statistics.

After many simulations, the tree's statistics — especially how many times each
root move was visited — tell you which move to play.

## 6.3 The heart of it: exploration vs. exploitation

Phase 1 (Selection) hides the whole cleverness of MCTS. At each node we must
choose a child, and we face the dilemma you met in
[Chapter 2](02-rl-fundamentals.md):

- **Exploitation:** pick the child that has looked best *so far*. But early on,
  "best so far" is based on very few samples and might be a fluke.
- **Exploration:** try children we haven't sampled much, in case they're
  secretly great.

Pick a rule that's too greedy and you'll lock onto the first move that got
lucky. Too explorative and you waste your budget spreading it thinly. We need a
principled balance.

### 📐 The UCT formula

Classic MCTS uses **UCT** (Upper Confidence bounds applied to Trees). For a
child, it computes:

$$\text{UCT} = \underbrace{\bar{Q}}_{\text{exploit}} \;+\; \underbrace{c \sqrt{\frac{\ln N_{\text{parent}}}{N_{\text{child}}}}}_{\text{explore}}$$

- $\bar{Q}$ is the child's average value so far (how good it has looked).
- $N_{\text{parent}}$, $N_{\text{child}}$ are visit counts.
- The second term is large when the child has few visits ($N_{\text{child}}$
  small) relative to how much we've searched overall ($N_{\text{parent}}$ large).
  So rarely-tried children get an exploration *bonus* that shrinks as we visit
  them.

Selection always picks the child with the **highest** UCT. Early on, the
exploration term dominates and MCTS samples broadly; as counts grow, $\bar{Q}$
takes over and the search commits to the genuinely strong moves.

## 6.4 What's wrong with random rollouts?

Classic MCTS's phase 3 estimates a leaf's value by **playing random moves to the
end** and using who won. This works for some games, but for chess it's terrible:
two random players produce a nonsensical game whose result tells you almost
nothing about whether the *leaf* position was actually good. You'd need
thousands of noisy rollouts to get a usable estimate.

This is the wall that classic game AI hit. AlphaZero knocked it down.

## 6.5 AlphaZero's upgrade: a neural network for evaluation

AlphaZero makes two changes, both powered by the two-headed neural network from
[Chapter 4](04-deep-rl-and-function-approximation.md) (and built in
[Chapter 10](10-the-neural-network.md)):

**Change 1 — replace the rollout with the value head.** Instead of playing to
the end, just *ask the network* how good the leaf is. The value head returns a
number in $[-1, +1]$ from the side-to-move's perspective. One forward pass
replaces an entire noisy game.

This is **bootstrapping**: we estimate a position's value from the network's
learned judgment rather than from a completed game. In our code, the leaf is
scored like this ([`mcts.py`](../src/chesszero/mcts.py)):

```python
if node.board.is_game_over():
    value = terminal_value(node.board)          # real result: +1 / 0 / -1
else:
    priors, value = evaluator.evaluate(node.board)   # the network's estimate
    _expand(node, priors)
```

> ⚠️ **This is why a small simulation budget still works.** Even if the search
> never reaches a checkmate, every leaf gets a value from the network. Low
> `num_simulations` means a *shallower, noisier* search — not *no* value. (Where
> does the network's judgment ultimately come from? Completed self-play games,
> which *do* run to the end — see [Chapter 5](05-self-play-and-games.md) and
> [Chapter 12](12-self-play-in-code.md).)

**Change 2 — replace UCT with PUCT, guided by the policy head.** The network's
policy head gives a **prior** probability for each move — its instant hunch about
which moves are worth considering. MCTS uses these priors to decide where to
spend its budget, so it never wastes simulations on moves the network thinks are
absurd. That's the next section.

## 6.6 📐 The PUCT selection rule

AlphaZero's selection rule is **PUCT** ("Predictor + UCT"). For each child of a
node, it scores:

$$\text{PUCT}(child) = Q(child) \;+\; c_{\text{puct}} \cdot P(child) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N_{\text{child}}}$$

Let's read every symbol, then look at the exact code.

| Symbol | Meaning | In the code |
| --- | --- | --- |
| $Q(child)$ | child's average value **from the parent's perspective** | `-child.value` |
| $P(child)$ | the network's prior probability for that move | `child.prior` |
| $N_{\text{parent}}$ | times the parent has been visited | `node.visit_count` |
| $N_{\text{child}}$ | times the child has been visited | `child.visit_count` |
| $c_{\text{puct}}$ | a constant trading off exploration vs. exploitation | `config.c_puct` (1.5) |

Here is the actual selection code — compare it line-for-line with the formula:

```python
def _select_child(node: Node, c_puct: float) -> Node:
    best_score = -float("inf")
    best_child = None
    sqrt_total = math.sqrt(node.visit_count)
    for child in node.children.values():
        # Child value is from the child's mover perspective (the opponent),
        # so the parent scores it as -child.value.
        q = -child.value
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    return best_child
```

The two terms pull in different directions:

- The **$Q$ term** exploits: it favors moves that have led to good outcomes.
- The **$U$ term** explores: it's large for moves with high prior $P$ and few
  visits $N_{\text{child}}$, and it shrinks as a move gets visited.

### 📐 Why the prior matters most at the start

Look at a brand-new child that has never been visited: $N_{\text{child}} = 0$, so
by the code in [Chapter 11](11-mcts-in-code.md) its `value` is defined as `0.0`,
hence $Q = 0$. Its whole score is just the $U$ term, which is proportional to its
**prior** $P$.

> ⚠️ **Consequence:** before any simulations have gathered evidence, MCTS
> explores moves in the order the *policy network* recommends. A strong policy
> head means the search spends its first, most valuable simulations on
> genuinely promising moves. This is the single biggest reason AlphaZero's search
> is so much more efficient than classic MCTS.

## 6.7 Averaging and sign-flipping: the backup

After evaluating a leaf, phase 4 sends the value back up the path. Two design
choices here are easy to get wrong, so we'll dwell on them. Here's the code:

```python
# Backup: alternate sign each ply back up the tree.
for path_node in reversed(search_path):
    path_node.visit_count += 1
    path_node.value_sum += value
    value = -value
```

**Why accumulate (`value_sum += value`) instead of overwriting?** Because a
node's value is a **Monte Carlo estimate** — an *average* over all the
simulations that passed through it. Each simulation is one noisy sample of "how
good is this node"; averaging many samples gives a stable estimate. A node's
reported value is:

```python
@property
def value(self) -> float:
    if self.visit_count == 0:
        return 0.0
    return self.value_sum / self.visit_count
```

The more a node is visited, the more accurate its value — and because PUCT sends
*more* visits to promising nodes, the best moves also get the *most accurately
estimated* values. That's a virtuous cycle.

**Why flip the sign each ply (`value = -value`)?** Because a value is always
expressed from the perspective of *the player to move* at that node, and the
players alternate. A leaf value of $+0.7$ ("great for the side to move at the
leaf") means $-0.7$ for the opponent one ply up, $+0.7$ two plies up, and so on.
Chess is **zero-sum**: what's good for me is exactly as bad for you (see
[Chapter 5](05-self-play-and-games.md)).

```
leaf: Black to move, value = +0.7   (good for Black)
  ▲   flip →  parent: White to move,      value = −0.7
  │   flip →  grandparent: Black to move,  value = +0.7
  │   flip →  root: White to move,         value = −0.7
```

This is exactly why selection uses `q = -child.value`: the parent evaluates a
child move by *negating* the child's value, because the child stores things from
the opponent's point of view. The two negations — in backup and in selection —
are consistent by design, and together they give MCTS a soft form of the minimax
logic from [Chapter 5](05-self-play-and-games.md).

## 6.8 The output: visit counts *are* the improved policy

Here's the beautiful payoff. After the search, we do **not** read off the child
with the best average value. We read off the child that was **visited most
often**.

Why visits and not value? Because PUCT already steered visits toward good moves
*and* kept them there only if they kept looking good under deeper search. The
visit distribution folds together prior, value, and the results of look-ahead
into one robust signal. A move visited 400 times out of 800 simulations is the
search's confident recommendation; a move visited twice was a passing glance.

We turn visit counts into a probability distribution $\pi$ over moves — the
**search policy**:

```python
def policy_from_visits(root: Node, temperature: float = 1.0) -> np.ndarray:
    ...
    scaled = visits ** (1.0 / temperature)
    scaled /= scaled.sum()
    ...
```

This $\pi$ is the single most important product of MCTS. In
[Chapter 7](07-the-alphazero-algorithm.md) you'll see it become the *training
target* for the policy head — the mechanism by which "thinking" teaches the
network to have better "instincts."

### Temperature: sample or commit?

The `temperature` parameter controls how we turn visits into a choice:

- **High temperature (τ = 1):** sample moves roughly in proportion to visits —
  encourages variety. Used for the opening moves of self-play games so the agent
  explores diverse positions.
- **Temperature → 0:** pick the single most-visited move, deterministically. Used
  for serious play and for the later moves of a game.

```python
def select_move(root: Node, temperature: float = 0.0) -> chess.Move:
    ...
    if temperature <= 1e-6:
        return moves[int(visits.argmax())]   # greedy: the most-visited move
    probs = visits ** (1.0 / temperature)
    probs /= probs.sum()
    return moves[int(np.random.choice(len(moves), p=probs))]   # sampled
```

## 6.9 Two knobs for exploration and strength

**Dirichlet noise at the root.** During self-play we don't want the agent playing
the *exact* same opening every game — it would never discover new ideas. So we
mix a little random noise into the root's priors, guaranteeing every root move
gets some exploration:

```python
def _add_dirichlet_noise(root: Node, config: Config) -> None:
    moves = list(root.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(moves))
    eps = config.dirichlet_epsilon
    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = (1 - eps) * child.prior + eps * float(n)
```

This noise is added **only at the root** and **only during self-play** (never
when playing for real) — it's for generating varied training data, which we'll
explore in [Chapter 12](12-self-play-in-code.md).

**Simulation count = strength.** There is no fixed search *depth*. Each
simulation deepens the tree by one node along whichever line PUCT favors, so
depth is *adaptive*: deep on forcing, promising lines and shallow on the rest.
More simulations means the tree concentrates deeper along the best lines and the
visit-count policy sharpens. This is the main dial between "fast and weak" and
"slow and strong."

## 6.10 MCTS simulates the opponent for free

One conceptual point that trips people up: MCTS needs **no external opponent**.
As the search descends the tree, every other ply is the opponent's turn — and
MCTS chooses the opponent's move using the *same* PUCT rule and the *same*
network. In other words, the search models the opponent as a copy of itself
playing its best, and the sign-flipping makes the arithmetic work out to
minimax. The "adversary" lives entirely inside the imagined tree.

This is different from **self-play** (Chapter 5 / 12), where the network makes
real moves for both sides across a whole game. Both use one network for both
sides — don't confuse the planning-time simulation (this chapter) with the
data-generation-time game (Chapter 12).

## 6.11 Putting it together

Here is one full MCTS search, start to finish, in words:

```
run_mcts(board):
    root ← new node for `board`
    priors, _ ← network.evaluate(board)      # policy head → priors
    expand root with those priors
    (self-play only) add Dirichlet noise to root priors

    repeat num_simulations times:
        # --- Selection ---
        node ← root
        while node is expanded:
            node ← child with highest PUCT score
        # --- Expansion + Evaluation ---
        if node is terminal:
            value ← real game result (from side-to-move view)
        else:
            priors, value ← network.evaluate(node)   # value head → estimate
            expand node with priors
        # --- Backpropagation ---
        for each node on the path, root-ward:
            node.visit_count += 1
            node.value_sum   += value
            value = -value                            # flip perspective

    return root   # its visit counts are the improved policy π
```

That's the engine. In [Chapter 11](11-mcts-in-code.md) we'll walk the real
implementation line by line — the `Node` class, lazy board materialization, and
every subtlety. But conceptually, you now understand the algorithm that turns a
fast, shallow network hunch into a slow, deep, reliable decision.

---

## Key takeaways

- The game tree is far too large to search fully, so MCTS spends a fixed
  **simulation budget** focused on promising lines, growing a lopsided tree.
- Each simulation has four phases: **selection, expansion, evaluation,
  backpropagation**.
- Classic MCTS evaluated leaves with **random rollouts**; AlphaZero replaces them
  with the network's **value head** (bootstrapping) — so even a small budget
  yields a value.
- **PUCT** guides selection using the network's **policy prior**; unvisited moves
  are ranked purely by their prior (since $Q = 0$), making a good policy head
  hugely valuable.
- Node values are **running averages** of backed-up leaf values, with the sign
  **flipped each ply** to respect the zero-sum, side-to-move convention.
- The search's real output is the **visit-count policy $\pi$** — the improved
  policy that will train the network in the next chapter.

## Exercises

1. In PUCT, what is the score of a child that has been visited zero times? Which
   quantity therefore decides which unvisited move gets explored first?
2. Explain, in one sentence each, *why* the backup **averages** values and *why*
   it **flips the sign** at each ply.
3. You run MCTS with `num_simulations = 4` on the opening position (35 legal
   moves). Will every legal move get visited? What does that imply about the
   quality of the resulting policy $\pi$? (See [Chapter 16](16-debugging-and-convergence.md).)
4. Why does AlphaZero read off the **most-visited** move rather than the move
   with the highest average value $Q$? What information does the visit count
   capture that a single $Q$ value does not?
5. Where in the four phases would classic MCTS have played a random game to the
   end, and what does our code do at that exact point instead? Quote the two-line
   `if/else` from [`mcts.py`](../src/chesszero/mcts.py).

---

> **Course:** [Home](README.md) · **Prev:** [5. Self-Play & Games](05-self-play-and-games.md) · **Next:** [7. The AlphaZero Algorithm](07-the-alphazero-algorithm.md)
