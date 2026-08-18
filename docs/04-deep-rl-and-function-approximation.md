# Chapter 4 — Deep RL & Function Approximation

> **Course:** [Home](README.md) · **Prev:** [3. MDPs & Value Functions](03-mdps-and-value-functions.md) · **Next:** [5. Games, Self-Play & Zero-Sum Search](05-self-play-and-games.md)

**What you'll learn**
- Why we *cannot* store a value for every chess position — the curse of dimensionality
- How a neural network becomes a *function approximator* for policies and values
- The three great families of RL algorithms: value-based, policy-based, and actor-critic
- Where AlphaZero sits — and why its use of **search** to generate training targets is unusual and important
- The exact loss functions we'll use later (MSE for value, cross-entropy for policy) and why

---

## 4.1 The wall we just hit

In [Chapter 3](03-mdps-and-value-functions.md) we defined the value function
$V(s)$ — how good it is to be in state $s$ — and saw that if we *knew* the value
of every state, playing well would be easy: just move toward the highest-valued
next state. We even sketched **value iteration**, an algorithm that fills in a
big table of $V(s)$ for every state $s$.

That algorithm is correct. It is also completely useless for chess.

Here's why. Value iteration stores one number per state in a lookup table:

```
   state s          V(s)
 ──────────────    ──────
 starting position   0.04
 1.e4                0.05
 1.e4 e5             0.03
 ...                  ...
```

How many rows would that table have? The number of legal chess positions is
estimated at around $10^{44}$. For comparison, there are about $10^{80}$ atoms
in the observable universe and only about $10^{17}$ seconds since the Big Bang.
You could not store this table, you could not fill it in, and you could not even
*visit* a meaningful fraction of its rows in the lifetime of the universe.

This is the **curse of dimensionality**: as the state space grows, table-based
methods become impossible. And chess is far from the worst case — Go is bigger,
and continuous-control robotics has *infinitely* many states.

⚠️ **The tabular methods from Chapter 3 are the theory; they are almost never the
implementation.** Real RL on interesting problems always replaces the table with
a *function*.

## 4.2 From a table to a function

Step back and notice something. A lookup table is just a function:

$$V : \text{state} \longrightarrow \text{number}$$

A table implements this function by *memorizing* every input-output pair. But we
have another way to implement a function that maps inputs to numbers and that
does **not** require storing every case: a **neural network**.

Instead of a table with $10^{44}$ rows, we use a network with, say, a few million
parameters $\theta$, and write:

$$V_\theta(s) \approx V(s)$$

The subscript $\theta$ means "this function is defined by these weights." We call
$V_\theta$ a **function approximator**, and when the approximator is a deep neural
network, the whole enterprise is called **Deep Reinforcement Learning**.

### Why this can possibly work: generalization

A table has no notion that two positions might be *similar*. Change one pawn and
it's a completely different row, learned from scratch. A neural network, by
contrast, **generalizes**: because it computes its output from features of the
input, positions that share features get similar evaluations *for free*, even
positions the network has never seen.

```
   TABLE                         NEURAL NETWORK
   ─────                         ──────────────
   memorizes each                learns features ("my king is
   position independently        exposed", "I'm up a rook") and
                                  reuses them across positions
   sees a new position           sees a new position →
   → no idea                     → similar features → sensible guess
```

This is the whole reason deep RL is feasible: the network trains on a
vanishingly small sample of positions (a few million from self-play) and
*generalizes* to the $10^{44}$ it will never see. Of course, generalization can
be *wrong* — a network can confidently misjudge a position — which is exactly why
we later add search on top (that's [Chapter 6](06-monte-carlo-tree-search.md)).

📐 **What does a network approximate here?** In our project the input $s$ is a
chess position encoded as a stack of $19$ planes of $8\times8$ (see
[Chapter 9](09-encoding-board-and-moves.md)), and the network is a convolutional
residual network — a natural fit for a grid-shaped input like a chessboard.

## 4.3 What can we approximate? Policies, values, or both

Recall two objects from Chapter 3:

- the **policy** $\pi(a \mid s)$ — a probability distribution over actions given a
  state ("what should I play here?"), and
- the **value** $V(s)$ or the action-value $Q(s,a)$ — a number estimating future
  reward ("how good is this?").

Both are functions of the state, so both can be approximated by a network. Which
one(s) you choose to approximate defines the three great families of RL
algorithms.

### Family 1 — Value-based methods

Approximate the **action-value** $Q_\theta(s,a)$: for a state, output a number for
each action estimating the reward you'll eventually get if you play it. Then act
**greedily** — pick the action with the highest $Q$:

$$a^\star = \arg\max_a Q_\theta(s, a)$$

The famous example is **DQN** (Deep Q-Network), which learned to play Atari games
from pixels. The policy is *implicit* — it's just "take the argmax of Q." Value
methods are sample-efficient and stable on discrete action spaces, but the argmax
gets awkward when there are millions of actions or continuous ones.

### Family 2 — Policy-based methods

Approximate the **policy** $\pi_\theta(a \mid s)$ directly — the network outputs
action probabilities — and adjust $\theta$ to make good actions more likely. The
core trick is the **policy gradient**: play some games, and for each action, nudge
its probability up if the game went better than expected and down if it went
worse. Schematically:

$$\theta \leftarrow \theta + \alpha \, (\text{return}) \, \nabla_\theta \log \pi_\theta(a \mid s)$$

"Make the actions that led to high return more probable." Policy methods handle
huge and continuous action spaces gracefully and can represent *stochastic*
policies, but raw policy gradients are noisy and can be slow.

### Family 3 — Actor-critic (both at once)

Approximate **both**: an *actor* $\pi_\theta$ that picks actions, and a *critic*
$V_\theta$ (or $Q_\theta$) that judges them. The critic's evaluation replaces the
noisy raw return in the policy-gradient update, dramatically reducing variance.
This "two heads" idea — one for the policy, one for the value — is exactly the
shape of the network we'll build.

```
                 ┌──────────────────────┐
   state  ─────► │   shared network body │
                 └──────────┬───────────┘
                     ┌──────┴───────┐
                     ▼              ▼
              policy head       value head
              π(a | s)            V(s)
             "what to play"     "how good is this"
```

## 4.4 Where AlphaZero sits — and what's unusual about it

Our project is **actor-critic in shape**: a single network with a **policy head**
and a **value head**. You can see this directly in
[`network.py`](../src/chesszero/network.py) — the `ChessNet.forward` method
returns a *pair*:

```python
def forward(self, x):
    x = self.stem(x)
    for block in self.res_blocks:
        x = block(x)

    # policy head → a score for every one of the 4672 possible moves
    p = F.relu(self.policy_bn(self.policy_conv(x)))
    p = p.reshape(p.size(0), -1)
    policy_logits = self.policy_fc(p)

    # value head → a single number in [-1, 1] for the whole position
    v = F.relu(self.value_bn(self.value_conv(x)))
    v = v.reshape(v.size(0), -1)
    v = F.relu(self.value_fc1(v))
    value = torch.tanh(self.value_fc2(v)).squeeze(-1)

    return policy_logits, value
```

So far, so actor-critic. But here is the twist that makes AlphaZero special, and
it's worth stating loudly:

> **AlphaZero does not train its policy with a policy gradient. It trains the
> policy to imitate the output of a *search*.**

In a classic policy-gradient method, the training signal for the policy comes
from the *returns* of games actually played. In AlphaZero, before every move we
run a **Monte Carlo Tree Search** ([Chapter 6](06-monte-carlo-tree-search.md))
that uses the current network to look ahead and think. The search produces a
*better* policy than the raw network — call it $\pi_{\text{MCTS}}$ — and a better
value estimate. We then train the network to match what the search concluded:

```
   raw network policy   ──(add lookahead via MCTS)──►   improved policy π_MCTS
          ▲                                                      │
          │                                                      │ training target
          └──────────────  train the network toward it  ◄────────┘
```

This is a form of **policy iteration** (Chapter 3): the search is a *policy
improvement operator*, and training is the *projection* of that improvement back
into the network's weights. We'll unpack exactly why this converges in
[Chapter 5](05-self-play-and-games.md) and assemble the full algorithm in
[Chapter 7](07-the-alphazero-algorithm.md). For now, just register the headline:
**the labels come from search, not from a gradient of the return.**

## 4.5 The two loss functions we'll train with

Once we have training targets — a target policy $\pi$ and a target value $z$ for
each position — training the network is ordinary supervised learning with two
losses, one per head. You already know both from basic ML.

### The value head → mean squared error (regression)

The value head outputs a single number in $[-1, 1]$ (via `tanh`), and we want it
to match the actual game outcome $z \in \{+1, 0, -1\}$ (win / draw / loss from
the side-to-move's perspective). That's a regression problem, so we use **mean
squared error**:

$$\mathcal{L}_{\text{value}} = \big(V_\theta(s) - z\big)^2$$

### The policy head → cross-entropy (distribution matching)

The policy head outputs a score for each of the 4672 possible moves; a softmax
turns those into a probability distribution. We want that distribution to match
the search policy $\pi$ (a target distribution over moves). Matching one
distribution to another is exactly what **cross-entropy** is for:

$$\mathcal{L}_{\text{policy}} = -\sum_a \pi(a)\,\log \big(\text{softmax}(V_\theta)\big)_a$$

### Added together

The total loss is simply their sum (plus a little weight decay to keep the
weights small):

$$\mathcal{L} = \underbrace{(V_\theta(s) - z)^2}_{\text{value: MSE}} \;+\; \underbrace{\Big(-\sum_a \pi(a)\log \pi_\theta(a\mid s)\Big)}_{\text{policy: cross-entropy}}$$

You will meet this exact expression in code in
[Chapter 13](13-training-and-replay-buffer.md), where
[`train.py`](../src/chesszero/train.py) computes
`F.mse_loss(...) + cross_entropy(...)`. Nothing exotic — the *cleverness of
AlphaZero is entirely in where the targets $z$ and $\pi$ come from*, not in the
losses that fit them.

## 4.6 One more ingredient: the replay buffer

There's a subtlety with training a network on data the network itself generates.
The positions within a single game are **highly correlated** — consecutive moves
are nearly identical positions. If you trained on them in order, each gradient
step would see a batch of near-duplicates, and the network would lurch around
overfitting to whatever it's currently playing.

The standard fix, inherited from DQN, is a **replay buffer**: a big pool that
holds the last $N$ training examples from *many* games. Each training step draws
a **random batch** from the pool, so a batch mixes positions from different games
and different points in time. This decorrelates the data and approximates the
"independent and identically distributed" samples that ordinary supervised
learning assumes.

```
  game 1 ─┐
  game 2 ─┼──►  ReplayBuffer  ──(random batch)──►  training step
  game 3 ─┘     (last N examples)                  (decorrelated)
```

We build this in [Chapter 13](13-training-and-replay-buffer.md)
([`replay_buffer.py`](../src/chesszero/replay_buffer.py)). For now, just know it
exists and why: **to break the correlation between consecutive self-play
positions.**

## 4.7 The estimate-vs-truth distinction (read this twice)

⚠️ The single most important conceptual point in this chapter: **the value the
network outputs is a learned *estimate*, not the truth.**

- $V_\theta(s)$ is the network's *guess* about how a position will turn out. At
  the start of training, when $\theta$ is random, this guess is pure noise.
- The only **ground truth** in the entire system is the actual outcome of a
  *finished game*: $z = +1, 0,$ or $-1$. That number is not an estimate — it's
  what really happened on the board.

The whole training process is the slow business of dragging the *estimate*
$V_\theta$ toward agreement with the *truth* $z$, position by position, game by
game. This distinction will matter enormously in [Chapter 5](05-self-play-and-games.md)
(where the ground-truth reward is what makes self-play work at all) and in
[Chapter 6](06-monte-carlo-tree-search.md) (where search uses the *estimate* to
avoid having to play every line to the end).

## 4.8 Where we are

We now know the shape of the solution: not a table but a **network** with two
heads (policy and value), trained with two ordinary losses (cross-entropy and
MSE) on decorrelated batches from a **replay buffer**. We also flagged the one
genuinely unusual thing about AlphaZero — its training targets come from a
**search**, not from a policy gradient.

The next chapter zooms into the setting that makes those search-based targets so
effective: **two-player, zero-sum games**, and the remarkable idea of a network
that improves by playing against itself.

---

## Key takeaways

- Chess has ~$10^{44}$ states, so a lookup table of values is impossible — we
  must **approximate** the value/policy functions with a neural network.
- A network **generalizes**: it learns features and reuses them, giving sensible
  guesses for positions it has never seen (a table cannot).
- The three RL families are **value-based** (learn $Q$, act greedily),
  **policy-based** (learn $\pi$, follow the policy gradient), and **actor-critic**
  (learn both). Our `ChessNet` is actor-critic in shape: a policy head + a value
  head.
- What makes AlphaZero unusual: it trains the policy to **imitate a search**, not
  via a policy gradient — a form of policy iteration.
- We fit the value head with **MSE** and the policy head with **cross-entropy**;
  a **replay buffer** decorrelates self-play data before training.
- The network's value is an **estimate**; only a finished game's outcome $z$ is
  **ground truth**.

## Exercises

1. Estimate how long it would take to *write down* a value table for chess if you
   could store one position per nanosecond. Compare to the age of the universe.
   Why does this immediately rule out tabular value iteration?
2. A network gives a position it has never seen a value of $+0.6$. Is that value
   "true"? What is the only quantity in this whole system that is not an estimate?
3. In [`network.py`](../src/chesszero/network.py), find the two heads. Which one
   ends in `tanh`, and why does that bound matter for the value?
4. Explain in one sentence, to a friend who knows supervised learning, why we
   need a replay buffer instead of just training on each game as we play it.
5. AlphaZero trains its policy toward the output of MCTS rather than with a policy
   gradient. In your own words, what is the "target" for the policy head, and
   where does it come from? (Full answer in [Chapter 7](07-the-alphazero-algorithm.md).)

---

> **Course:** [Home](README.md) · **Prev:** [3. MDPs & Value Functions](03-mdps-and-value-functions.md) · **Next:** [5. Games, Self-Play & Zero-Sum Search](05-self-play-and-games.md)
