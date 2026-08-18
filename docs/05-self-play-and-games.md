# Chapter 5 — Games, Self-Play & Zero-Sum Search

> **Course:** [Home](README.md) · **Prev:** [4. Deep RL & Function Approximation](04-deep-rl-and-function-approximation.md) · **Next:** [6. Monte Carlo Tree Search](06-monte-carlo-tree-search.md)

**What you'll learn**
- What makes a game **two-player, zero-sum**, and why that structure is a gift
- The game tree and **minimax** — the classical idea of optimal play
- Why a value is always "from the side-to-move's perspective," forcing a **sign flip** between plies
- **Self-play** as the engine of learning: one network, both sides
- Why self-play *converges* even though the agent only ever faces itself
- The exploration you must add — **Dirichlet noise** and **temperature** — and the failure mode if you don't

---

## 5.1 Two-player, zero-sum: your win is my loss

Everything so far (Chapters 2–4) described a *single* agent in an environment. A
game adds a second agent — an opponent — and that changes the picture in a way
that turns out to be enormously helpful.

Chess is a **two-player, zero-sum** game:

- **Two-player:** White and Black alternate moves.
- **Zero-sum:** the players' rewards are exact opposites. If White wins ($+1$),
  Black loses ($-1$); a draw is $0$ for both. There is no outcome where both do
  well or both do badly. The rewards sum to zero — hence "zero-sum."

$$r_{\text{White}} + r_{\text{Black}} = 0$$

Why is this a gift? Because it means **the opponent is not a separate problem to
model.** In a general multi-agent setting you might have to guess the other
player's goals. In a zero-sum game the opponent's goal is, by definition, exactly
the negation of yours. That single fact lets one network play *both* sides — the
foundation of self-play (§5.4) — and it lets a search account for the opponent by
just flipping a sign (§5.3).

## 5.2 The game tree and minimax

Imagine writing out every possible continuation of a position as a branching
tree. The root is the current position. Its children are the positions after each
legal move. Their children are the positions after every reply. And so on, down
to leaves where the game is over (checkmate, stalemate, draw).

```
                    (White to move)
                    /     |      \
                  e4      d4      Nf3        ← White's choices
                 / \     / \      / \
               ...  ... ...  ... ...  ...    ← Black's replies
                                    ...
                              checkmate / draw   ← leaves have real results
```

This is the **game tree**. If it were small enough to write out completely,
optimal play would be a solved problem via **minimax**:

- At a leaf, you know the true result ($+1$, $0$, $-1$).
- At a node where **you** move, you'll pick the child that *maximizes* your
  result.
- At a node where your **opponent** moves, they'll pick the child that
  *minimizes* your result (because zero-sum: minimizing yours maximizes theirs).

Propagate those choices up from the leaves and the root gets its true value:
the outcome of the game under perfect play by both sides. That alternating
`max`/`min` is where "minimax" gets its name.

Minimax is *correct* and *completely impractical* for chess — the tree has more
leaves than we could ever enumerate (the same $10^{44}$ wall from
[Chapter 4](04-deep-rl-and-function-approximation.md)). The rest of this course
is about approximating minimax without building the whole tree:
[Monte Carlo Tree Search](06-monte-carlo-tree-search.md) explores only the
promising branches, and the value network guesses the result of leaves we never
reach.

## 5.3 Whose perspective? The sign-flip convention

Here is a small idea that causes big confusion if you skip it, so we'll be
explicit.

A value like "+0.7" is meaningless until you say *for whom*. We adopt one clean
convention, used everywhere in the code:

> **A value is always expressed from the perspective of the player who is about
> to move in that position.**

So $V(s) = +0.7$ means "the side to move in $s$ is doing well," whoever that is.
This is convenient — the network doesn't need to know or care whether it's
"playing White" — but it has a consequence. When you step from a position to the
position *after* a move, the side to move flips (it's now the opponent's turn).
The very same real-world outcome therefore flips sign between the two positions:

```
  position s        move       position s'
  (Black to move)  ──────►     (White to move)
  value +0.7                   value -0.7
  "great for Black"            "bad for White" ... same reality
```

You can see this convention in two places in
[`mcts.py`](../src/chesszero/mcts.py). First, when a game ends, `terminal_value`
returns the result *from the side-to-move's view*:

```python
def terminal_value(board):
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    # if it's a win, the side to move was the one checkmated → -1 for them
    return 1.0 if outcome.winner == board.turn else -1.0
```

Second, when a search carries a value back up the tree, it **flips the sign at
every ply**, and when a parent scores a child it *negates* the child's value:

```python
# scoring a child during selection:
q = -child.value          # child's value is from the opponent's view

# carrying a leaf value back up the search path:
for path_node in reversed(search_path):
    path_node.visit_count += 1
    path_node.value_sum += value
    value = -value        # flip perspective at each ply
```

📐 **Why `value = -value` is exactly right.** If a leaf is worth $+0.7$ to the
player moving there, it is worth $-0.7$ to that player's parent (the opponent),
$+0.7$ to the grandparent, and so on — alternating. Flipping the sign each step
keeps every node's stored value in *its own* mover's perspective. This is
minimax's `max`/`min` alternation in disguise: because each node negates its
children and then takes the best, "maximize my negated-opponent-value" is the
same as "opponent minimizes my value." We'll use this constantly in
[Chapter 11](11-mcts-in-code.md).

## 5.4 Self-play: one network, both sides

Now the central idea. To learn, we need games. Where do the games come from?

In AlphaZero, from the agent playing **itself**. A single network sits at the
board and chooses moves for *both* White and Black. Because a value is
perspective-relative (§5.3), the same network is perfectly happy to move for
either color — it always just answers "what's good for the side to move?"

In our project this is [`selfplay.py`](../src/chesszero/selfplay.py). The heart
of it is `play_game`, and stripped to essentials it's a loop:

```python
board = chess.Board()          # standard starting position
while not board.is_game_over() and move_number < config.max_moves:
    root = run_mcts(board, evaluator, config, add_noise=True)  # think
    policy = policy_from_visits(root)                          # record what search wanted
    history.append((encode_board(board), policy, board.turn))
    move = select_move(root, temperature=temperature)          # choose a move
    board.push(move)                                           # play it
```

One board, one network, alternating turns until the game ends. When it does, we
compute the result and hand every stored position its label:

```python
result = _game_result(board)      # +1 white win, -1 black win, 0 draw
for state, policy, side_to_move in history:
    value = result if side_to_move == chess.WHITE else -result   # perspective!
    examples.append(Example(state=state, policy=policy, value=float(value)))
```

Notice the perspective flip again: a position where **Black** was to move gets
`-result`, so that "value" always means "good for the player who was about to
move here." Those `Example`s are the training data for the network
([Chapter 13](13-training-and-replay-buffer.md)).

## 5.5 The objection — and why self-play actually works

Here is the objection almost everyone raises, and it is a good one:

> "The agent only ever plays against *itself*, and early on it's terrible. How
> can you learn to play well by playing a weak opponent? Won't it just get good
> at beating bad players and plateau?"

⚠️ This intuition is wrong, and understanding *why* is the key to understanding
AlphaZero. There are three reasons self-play climbs from random to superhuman.

### Reason 1 — The reward is ground truth, not relative skill

When two weak players stumble through a game and one delivers checkmate, that
checkmate is a **real win under the rules of chess**, no matter how badly both
played. The label $z = +1$ is *correct*, not "correct relative to a weak
opponent." You are not learning "how to beat this particular weak player"; you're
learning "these positions lead to real wins and real losses." The ground-truth
outcome ([Chapter 4](04-deep-rl-and-function-approximation.md), §4.7) doesn't
care about the skill level of the players who produced it.

### Reason 2 — Search is a policy-improvement operator

This is the engine. Before each move, MCTS looks ahead using the current network
and produces a policy $\pi_{\text{MCTS}}$ that is **better than the network's raw
policy** — it's the network's own judgment, sharpened by lookahead. When we then
train the network toward $\pi_{\text{MCTS}}$, we are pulling it toward something
*strictly stronger than its current self*:

```
   current network  ──MCTS(lookahead)──►  stronger policy π_MCTS
         ▲                                          │
         └────────── train net toward π_MCTS ◄──────┘
                  (net becomes a bit stronger; repeat)
```

This is exactly **policy iteration** from [Chapter 3](03-mdps-and-value-functions.md):
*improve* the policy (via search), *evaluate/fit* it (via training), repeat. Each
turn of the loop the "teacher" (search) is a better version of the "student"
(the network), and the student chases it. There is no fixed weak opponent to
plateau against — the bar rises with you.

### Reason 3 — Self-play is an automatic curriculum

Think about how hard it is to learn against a fixed opponent. Against a
grandmaster, a beginner loses every game and gets almost no useful signal (every
action looks equally doomed). Against a random mover, you win every game and
learn nothing about real chess. The *ideal* opponent is one exactly at your own
level — challenging but beatable — so that the difference between your good and
bad moves actually shows up in the results.

Self-play hands you that ideal opponent for free: it is **always exactly your own
strength**, because it *is* you. As you improve, so does your opponent, in
lockstep. The curriculum tunes itself.

### Putting it together

$$\underbrace{\text{ground-truth reward}}_{\text{correct signal}} \;+\; \underbrace{\text{search} > \text{net}}_{\text{always improving}} \;+\; \underbrace{\text{matched opponent}}_{\text{ideal difficulty}} \;\Longrightarrow\; \text{climb from random to strong}$$

The classic objection assumes you're trying to *beat a fixed weak player*. You're
not. You're trying to be **better than your former self**, measured against an
**objective** win/loss signal, with a teacher (search) that is always a step
ahead. That is why it works.

## 5.6 The cold start: where does the first signal come from?

At iteration 0 the network is random. Its value guesses are noise and its policy
is nearly uniform. So on the very first games, is there *any* signal?

Yes — but only one kind: the **outcome of completed games**. Even random-ish play
eventually produces checkmates, stalemates, and draws, and those results are real
($z \in \{+1,0,-1\}$). Training the value head on them gives it its first faint
grip on reality ("positions like this tend to be lost"). Once the value head is
even slightly better than noise, the search built on it becomes slightly
purposeful, which produces slightly better games, which trains a slightly better
network. This is **bootstrapping**: the system pulls itself up by using its own
(improving) estimates. We'll see in [Chapter 6](06-monte-carlo-tree-search.md)
that the search never has to play a line to the end precisely because it can lean
on the value estimate — but the estimate is only meaningful *because* completed
games keep grounding it in truth.

## 5.7 Exploration: don't play the same game every time

There's a danger lurking in §5.4. If the network always plays its single
best-looking move, self-play becomes nearly deterministic: the agent plays *the
same game* over and over, sees the same handful of positions, and never discovers
the alternatives that might be better. Learning stalls. We must inject
**exploration** — deliberate variety — into self-play.

AlphaZero uses two tools, both present in our code and both revisited in detail
in [Chapter 6](06-monte-carlo-tree-search.md) and [Chapter 12](12-self-play-in-code.md).

**Dirichlet noise at the search root.** Before searching a self-play position, we
perturb the network's move priors at the *root* with random noise, so that even
moves the network currently dislikes get *some* chance to be explored:

```python
def _add_dirichlet_noise(root, config):
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(moves))
    eps = config.dirichlet_epsilon
    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = (1 - eps) * child.prior + eps * float(n)
```

This is applied *only* during self-play (`add_noise=True` in
`run_mcts`), never when the trained agent plays for real — you want exploration
while learning, best-effort when performing.

**Temperature on move selection.** For the first several moves of each game, we
*sample* a move in proportion to how much the search visited it, rather than
always taking the most-visited one. That's the `temperature` in `select_move`:

```python
temperature = 1.0 if move_number < config.temperature_moves else 0.0
move = select_move(root, temperature=temperature)
```

High temperature early ⇒ varied openings and a diverse dataset. Temperature $0$
later ⇒ sharp, best-effort play once it matters. (The mechanics of `temperature`
and visit counts are [Chapter 6](06-monte-carlo-tree-search.md).)

## 5.8 The failure mode: games that never end decisively

⚠️ Self-play only teaches the value head if games actually *end with a result*.
Imagine games that almost always hit the move limit and get scored as draws. Then
every label is $z = 0$, the value head learns "everything is a draw," the gradient
signal vanishes, and the agent never learns to *win*.

Two things guard against this, and both are tunable:

- **The rules themselves force termination.** Chess's fivefold-repetition and
  seventy-five-move rules make `board.is_game_over()` eventually true, and our
  loop also has a hard `max_moves` cap as a backstop.
- **Exploration (§5.7) keeps games varied and decisive.** Enough search
  simulations and enough move-cap headroom let the agent actually convert winning
  positions instead of shuffling into repetitions.

If you ever see training stall with everything drawing, this is the first place
to look — we return to it as a concrete debugging story in
[Chapter 16](16-debugging-and-convergence.md).

## 5.9 Where we are

We now understand the *setting* that makes AlphaZero tick: a two-player zero-sum
game, where a perspective-relative value and a sign-flip let one network reason
about both sides, and where **self-play** produces an endless stream of
ground-truth-labelled games against a perfectly matched opponent. We've argued —
carefully — why this converges, and what exploration you must add so it doesn't
get stuck.

We've mentioned "the search" a dozen times without saying how it works. That's
the next chapter: **Monte Carlo Tree Search**, the algorithm that turns the
network's fast intuition into a slow, strong decision.

---

## Key takeaways

- Chess is **two-player, zero-sum**: your reward is exactly the negation of your
  opponent's, which lets one network reason about both sides.
- **Minimax** defines optimal play on the game tree (`max` on your turn, `min` on
  the opponent's), but the tree is far too big to enumerate.
- Values are stored **from the side-to-move's perspective**, so passing a value
  between plies flips its sign — this is `value = -value` and `q = -child.value`
  in [`mcts.py`](../src/chesszero/mcts.py).
- **Self-play** (`selfplay.play_game`) has one network play both sides; finished
  games are labelled with the outcome and become training data.
- Self-play converges for three reasons: the reward is **ground truth**, **search
  is a policy-improvement operator** (policy iteration), and it's an **automatic
  curriculum**. The "it only plays weak opponents" objection misunderstands the
  goal.
- Add **Dirichlet noise** (root) and **temperature** (early moves) for
  exploration, or self-play stagnates; watch out for the **all-draws** failure
  mode.

## Exercises

1. A position has value $-0.4$ from the side-to-move's perspective. What is its
   value from the *other* player's perspective? Which line in
   [`mcts.py`](../src/chesszero/mcts.py) performs this conversion during back-up?
2. Explain the sentence "self-play is an automatic curriculum" to someone who
   thinks you can only learn by playing stronger opponents. Where does their
   intuition come from, and why doesn't it apply here?
3. In `_game_result`/`play_game`, why does a position where Black was to move get
   labelled with `-result` instead of `result`?
4. Suppose you set `temperature_moves = 0` (never sample, always take the top
   move). Predict what happens to the *diversity* of self-play games and to
   learning. Then predict the effect of removing the Dirichlet noise.
5. You run training and every single game is a draw at the move limit. Using
   §5.8, list two hyperparameters you'd change first and say why.

---

> **Course:** [Home](README.md) · **Prev:** [4. Deep RL & Function Approximation](04-deep-rl-and-function-approximation.md) · **Next:** [6. Monte Carlo Tree Search](06-monte-carlo-tree-search.md)
