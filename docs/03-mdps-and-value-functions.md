# Chapter 3 — MDPs & Value Functions

> **Course:** [Home](README.md) · **Prev:** [2. RL Fundamentals](02-rl-fundamentals.md) · **Next:** [4. Deep RL & Function Approximation](04-deep-rl-and-function-approximation.md)

**What you'll learn**
- The **Markov property** and the formal **Markov Decision Process** (MDP)
- Why chess is (almost) Markovian, and how our 19-plane encoding *makes* it so
- Formal definitions of the state-value $V^\pi$ and action-value $Q^\pi$
- The **Bellman equations** — the recursive heartbeat of all of RL
- Optimal value functions and the **Bellman optimality equations**
- **Policy iteration** and the **policy improvement theorem** — the bridge to MCTS
- Why exact, tabular methods cannot possibly scale to chess

---

## 3.1 The Markov property

[Chapter 2](02-rl-fundamentals.md) gave us states, actions, rewards, returns, and
policies. To reason about them cleanly we need one more assumption — a very
powerful one — called the **Markov property**:

> The future depends only on the *present* state, not on the *history* of how you
> got there.

Formally, the probability of the next state and reward depends only on the current
state and action:

$$\Pr(s_{t+1}, r_{t+1} \mid s_t, a_t) = \Pr(s_{t+1}, r_{t+1} \mid s_0, a_0, \dots, s_t, a_t)$$

If this holds, then the current state is a **sufficient statistic** for
decision-making: you lose nothing by forgetting the entire past and keeping only
$s_t$. This is what makes value functions well-defined — "the value of a state"
only makes sense if the state alone determines what can happen next.

## 3.2 The Markov Decision Process

A problem that satisfies the Markov property is a **Markov Decision Process
(MDP)** — the mathematical object that *all* of RL is built on. An MDP is a tuple:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

| Symbol | Name | In chess |
| --- | --- | --- |
| $\mathcal{S}$ | set of **states** | all legal board positions |
| $\mathcal{A}$ | set of **actions** | all moves (our fixed 4672-action space) |
| $P(s' \mid s, a)$ | **transition** dynamics | the rules: which position follows a move |
| $R(s, a)$ | **reward** function | 0 mid-game; ±1 at a decisive end; 0 for a draw |
| $\gamma$ | **discount** factor | $\approx 1$ (sparse terminal reward) |

Two remarks specific to chess:

- **The dynamics $P$ are deterministic *and* known.** Given a position and a
  legal move, the next position is fixed — no dice, no randomness. And we *have*
  $P$: it's the rulebook, implemented by `python-chess`. This is precisely the
  "perfect model" from §2.10 that lets us plan with search.
- **It's a *two-player* MDP.** Strictly, chess is a two-agent zero-sum game, not a
  one-agent MDP. The standard trick — which our code uses — is to view it from the
  perspective of *the player to move*, flipping the sign of value at each ply so
  the opponent's gain is our loss. [Chapter 5](05-self-play-and-games.md) makes
  this rigorous; for this chapter, imagine a single decision-maker.

## 3.3 Is chess really Markovian? (and how the encoding ensures it)

Here's a subtlety that most tutorials skip but which is baked right into our code.
Is a chess *board diagram* — just the pieces — a Markov state? **No, not quite.**
The legal moves available (and therefore the future) depend on more than piece
placement:

- **Castling rights**: whether you may castle depends on whether the king or rook
  has *ever* moved — a fact about history, not visible in the piece positions.
- **En passant**: the special pawn capture is only legal *immediately* after the
  opponent's two-square pawn advance — a fact about the previous move.
- **The fifty-move / halfmove clock**: a draw can be claimed after 50 moves with
  no capture or pawn move — a running count, again about history.
- **Whose turn it is**: obviously part of the state.

If we fed the network only the pieces, the state would *not* be Markovian: two
identical-looking boards could have different legal moves and different futures.
The fix is to fold all of that "hidden history" *into the state itself*, so the
state once again fully determines the future. That is exactly what our encoding
does. Look at the plane layout in
[`encoding.py`](../src/chesszero/encoding.py):

```python
# Plane layout (all 8x8):
#   0-5   : white pieces  (P, N, B, R, Q, K)
#   6-11  : black pieces  (P, N, B, R, Q, K)
#   12    : side to move  (all ones if white to move)
#   13-16 : castling rights (W kingside, W queenside, B kingside, B queenside)
#   17    : en-passant target square
#   18    : halfmove clock, normalized by 100
INPUT_PLANES = 19
```

Planes 0–11 are the pieces. But planes 12–18 are the "make-it-Markov" planes:
side to move, castling rights, en-passant target, and the halfmove clock. By
including them, [`encode_board`](../src/chesszero/encoding.py) produces a state
representation that **restores the Markov property** — the `(19, 8, 8)` tensor
contains everything needed to determine the legal moves and the future.

> 📐 **The one approximation.** True chess draw rules also involve *threefold
> repetition* (the exact position occurring three times), which technically
> depends on the full move history. Our single-position encoding doesn't track
> repetition counts, so the state is *approximately* Markov — a deliberate,
> minor simplification. See [Chapter 9](09-encoding-board-and-moves.md) and
> [Chapter 17](17-scaling-and-improvements.md) for the consequences and fixes.

## 3.4 The return, once more

Recall the **return** from §2.5 — cumulative discounted reward from time $t$:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$$

A tiny but useful algebraic fact: the return is **recursive**. Peel off the first
term:

$$G_t = r_{t+1} + \gamma\big(r_{t+2} + \gamma r_{t+3} + \cdots\big) = r_{t+1} + \gamma\, G_{t+1}$$

"My return now = the reward I get next + (discounted) my return from the next
state." This innocuous line is the seed of the **Bellman equations** below, and
ultimately of the value back-ups performed by MCTS in
[Chapter 11](11-mcts-in-code.md).

## 3.5 Value functions, formally

In §2.8 we described value intuitively. Now the definitions.

The **state-value function** of a policy $\pi$ is the expected return starting
from state $s$ and following $\pi$ thereafter:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\big[\, G_t \mid s_t = s \,\big]$$

The **action-value function** (or **Q-function**) is the expected return starting
from $s$, taking action $a$ first, then following $\pi$:

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\big[\, G_t \mid s_t = s,\, a_t = a \,\big]$$

The relationship between them: the value of a state is the average of the
action-values over whatever moves the policy would play,

$$V^{\pi}(s) = \sum_{a} \pi(a \mid s)\, Q^{\pi}(s, a).$$

In chess terms: $V$ = "how good is this position for me?", $Q$ = "how good is this
*move*?". The value head of [`ChessNet`](../src/chesszero/network.py) is our
learned approximation of $V$, and — as we'll see — the *visit counts* produced by
MCTS give us something very much like a $Q$-based ranking of moves.

## 3.6 The Bellman expectation equations 📐

Combine the recursive return $G_t = r_{t+1} + \gamma G_{t+1}$ with the definition
of value, and you get the celebrated **Bellman expectation equation** for $V^\pi$:

$$V^{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\Big[ R(s,a) + \gamma\, V^{\pi}(s') \Big]$$

Read it right to left as a story:

1. From state $s$, the policy chooses action $a$ with probability $\pi(a\mid s)$.
2. The environment lands you in $s'$ with probability $P(s'\mid s,a)$, paying
   reward $R(s,a)$.
3. From $s'$ onward you collect $V^\pi(s')$, discounted by $\gamma$.

Average over all the branches and you recover $V^\pi(s)$. The analogous equation
for $Q$ is:

$$Q^{\pi}(s, a) = \sum_{s'} P(s'\mid s, a)\Big[ R(s,a) + \gamma \sum_{a'} \pi(a'\mid s')\, Q^{\pi}(s', a') \Big]$$

The essential idea to carry forward: **the value of a state is defined in terms of
the values of its successor states.** Value estimates lean on other value
estimates. This "bootstrapping" is exactly what MCTS does when it backs a leaf
evaluation up the tree ([Chapter 11](11-mcts-in-code.md)).

## 3.7 Optimal value functions

We don't ultimately want the value of *some* policy — we want the *best possible*
play. Define the **optimal** value functions as the best achievable over all
policies:

$$V^{*}(s) = \max_{\pi} V^{\pi}(s), \qquad Q^{*}(s, a) = \max_{\pi} Q^{\pi}(s, a).$$

These satisfy the **Bellman optimality equations**, where instead of *averaging*
over the policy's actions we *maximize* over actions (a rational agent will pick
the best one):

$$V^{*}(s) = \max_{a} \sum_{s'} P(s'\mid s, a)\Big[ R(s,a) + \gamma\, V^{*}(s') \Big]$$

$$Q^{*}(s, a) = \sum_{s'} P(s'\mid s, a)\Big[ R(s,a) + \gamma \max_{a'} Q^{*}(s', a') \Big]$$

If you knew $Q^{*}$, playing perfectly would be trivial: in every state, take
$\arg\max_a Q^{*}(s,a)$. The entire difficulty of RL is that we *don't* know these
functions and must approximate them from experience.

> 📐 **Why "max" is where the game lives.** In a two-player game the opponent is
> also maximizing *their* value, i.e. minimizing yours. Replacing that averaging
> with alternating maxima/minima is classical **minimax**. MCTS is a soft,
> sampled version of minimax — it explores the promising branches instead of the
> whole tree. More in [Chapter 5](05-self-play-and-games.md) and
> [Chapter 6](06-monte-carlo-tree-search.md).

## 3.8 Policy iteration and the policy improvement theorem

How do we actually *reach* an optimal policy? The classical answer is a loop
called **generalized policy iteration**, which alternates two steps:

```
        ┌────────────────────────────┐
        │   POLICY EVALUATION         │   estimate V^π for the current policy π
        │   "how good is this policy?"│
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   POLICY IMPROVEMENT        │   make π greedier w.r.t. V^π
        │   "act better given values" │
        └────────────┬───────────────┘
                     │
                     └──────────► repeat until π stops changing (→ π*)
```

The reason this works is the **policy improvement theorem**:

> If you build a new policy $\pi'$ that, in every state, picks the action with the
> highest $Q^{\pi}(s, a)$ under the *old* policy's values, then $\pi'$ is at least
> as good as $\pi$ everywhere: $V^{\pi'}(s) \ge V^{\pi}(s)$ for all $s$.

In words: *acting greedily with respect to your current value estimates can only
help.* Iterate — evaluate, improve, evaluate, improve — and you climb monotonically
toward the optimum $\pi^{*}$.

**This is the single most important idea to carry into Part II.** AlphaZero is
generalized policy iteration in disguise:

- **Policy evaluation** ≈ the network learns to predict outcomes and good moves
  from self-play data.
- **Policy improvement** ≈ **MCTS**. Running a search on top of the current
  network produces a policy that is *provably better than the network alone* —
  exactly the improvement step. We then train the network to imitate it, and
  repeat.

We will name this connection explicitly and cash it out in code in
[Chapter 6](06-monte-carlo-tree-search.md) and [Chapter 7](07-the-alphazero-algorithm.md).
Whenever you see "MCTS is a policy-improvement operator," this theorem is why.

## 3.9 Why exact methods can't touch chess

Classical policy iteration and its cousin **value iteration** work by storing a
number for *every* state (a big table) and sweeping through them repeatedly. For
small problems — a gridworld with 100 squares — this is perfect and provably
optimal.

Now count chess. The number of legal chess positions is estimated at more than
$10^{40}$. To put that in perspective:

```
gridworld states                         ~10^2   ✅ store them all in a table
Tic-tac-toe states                       ~10^3   ✅ trivial
Backgammon states                        ~10^20  ⚠️  already too many for a table
Chess states                             ~10^44  ❌ hopeless — more than atoms on Earth
Go states                                ~10^170 ❌❌
```

You cannot store one number per state. You cannot even *visit* each state once —
not in the lifetime of the universe. Tabular value/policy iteration is dead on
arrival.

The escape is **generalization**: instead of a lookup table $V(s)$, learn a
*function* $V_\theta(s)$ — parameterized by weights $\theta$ — that takes a state
and *computes* a value, and that generalizes from the positions it has seen to
the astronomically many it hasn't. That function is a **neural network**, and how
we build and train it is the subject of the next chapter.

---

## Key takeaways

- The **Markov property** says the future depends only on the present state; an
  environment satisfying it is a **Markov Decision Process** $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.
- A raw chess diagram is *not* Markov; our 19-plane encoding restores it by adding
  **side-to-move, castling rights, en-passant, and the halfmove clock** as planes.
- $V^{\pi}(s)$ scores positions, $Q^{\pi}(s,a)$ scores moves; the **Bellman
  equations** define each value recursively in terms of successor values
  (bootstrapping).
- The **Bellman optimality equations** replace averaging with maximizing; knowing
  $Q^{*}$ would make perfect play a simple $\arg\max$.
- **Policy iteration** (evaluate ⇄ improve) climbs to the optimum, justified by the
  **policy improvement theorem** — and AlphaZero is this loop with **MCTS as the
  improvement operator**.
- Exact tabular methods are impossible for chess's $10^{40+}$ states, forcing us to
  **approximate** value and policy with a neural network — the topic of
  [Chapter 4](04-deep-rl-and-function-approximation.md).

## Exercises

1. Give a concrete example of two chess positions with *identical piece placement*
   but *different legal moves*. Which of the encoding planes 12–18 distinguishes
   them?
2. Starting from $G_t = r_{t+1} + \gamma G_{t+1}$, and using
   $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, sketch the two lines of algebra
   that give the Bellman expectation equation for $V^\pi$.
3. In one sentence each, state policy *evaluation* and policy *improvement*. Which
   one will MCTS play the role of?
4. The action-value relationship is
   $V^\pi(s) = \sum_a \pi(a\mid s)\,Q^\pi(s,a)$. If $\pi$ is *deterministic*
   (always plays one move $a^\*$), what does this reduce to?
5. Estimate how much memory a tabular $V(s)$ for chess would need if each value is
   a 4-byte float and there are $10^{44}$ states. Compare to the number of atoms
   in your body ($\sim 10^{27}$). What does this tell you about the need for
   function approximation?

---

> **Course:** [Home](README.md) · **Prev:** [2. RL Fundamentals](02-rl-fundamentals.md) · **Next:** [4. Deep RL & Function Approximation](04-deep-rl-and-function-approximation.md)
