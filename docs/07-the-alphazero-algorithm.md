# Chapter 7 — The AlphaZero Algorithm

> **Course:** [Home](README.md) · **Prev:** [6. Monte Carlo Tree Search](06-monte-carlo-tree-search.md) · **Next:** [8. Project Setup & Architecture](08-project-setup-and-architecture.md)

**What you'll learn**
- How the two-headed network and MCTS combine into one self-improving system
- Why MCTS is a **policy-improvement operator**, and the loop as **generalized policy iteration**
- The exact training targets ($\pi$ and $z$) and the AlphaZero **loss function**
- The full algorithm in boxed pseudocode
- How AlphaZero differs from AlphaGo — and where each idea lives in our code

---

## 7.1 The whole idea in one paragraph

We have, from the last four chapters, three ingredients:

1. A **neural network** $f_\theta$ that, given a position $s$, instantly outputs a
   **policy** $p$ (a hunch about good moves) and a **value** $v$ (a hunch about
   who's winning) — [Chapter 4](04-deep-rl-and-function-approximation.md).
2. **MCTS**, which uses $f_\theta$ to *think* — turning the fast, shallow hunch
   into a slow, deep, reliable **search policy** $\pi$ and search value —
   [Chapter 6](06-monte-carlo-tree-search.md).
3. **Self-play**, which uses the current agent to generate games and, from their
   outcomes, ground-truth reward — [Chapter 5](05-self-play-and-games.md).

AlphaZero ties them into a loop:

> Use the network to power a search. The search plays better than the raw network.
> Record what the search did and how the games turned out. Train the network to
> imitate the search and predict the outcomes. Now the network is better — so the
> search it powers is better too. Repeat.

That's it. The rest of this chapter makes each step precise.

## 7.2 The two-headed network $f_\theta$

Formally, the network is a function of the parameters $\theta$:

$$f_\theta(s) = (\mathbf{p}, v)$$

- $\mathbf{p}$ is a vector of **prior** move probabilities (after masking to legal
  moves — [Chapter 9](09-encoding-board-and-moves.md)). It is the network's
  *policy*: "which moves look worth trying?"
- $v \in [-1, +1]$ is a single scalar, the network's *value*: "how good is this
  position for the side to move?"

Both heads share a common body (a residual tower), so they build on the same
internal understanding of the position. You'll build $f_\theta$ in
[Chapter 10](10-the-neural-network.md).

> ⚠️ **Two heads, two roles.** The policy head scores *moves* (one number per
> move). The value head scores the *position* (one number total). Don't conflate
> them — the value is not "the probability of the best move."

## 7.3 The key move: search improves the policy

Here is the theoretical crux of AlphaZero, and the reason it converges.

Recall the **policy improvement** idea from
[Chapter 3](03-mdps-and-value-functions.md): if you have a policy and its value
estimates, you can construct a *strictly better* policy by acting greedily with
respect to those values. MCTS is a concrete, powerful realization of exactly
that principle:

$$\pi = \text{MCTS}(s, f_\theta) \quad\text{is a better policy than}\quad \mathbf{p} = f_\theta(s).$$

Why is $\pi$ better than $\mathbf{p}$? Because $\mathbf{p}$ is a snap judgment,
while $\pi$ is the result of *looking ahead*: MCTS took the network's own priors
and values and refined them through hundreds of simulated lines, keeping moves
only if they held up under deeper scrutiny (Chapter 6). The visit-count
distribution $\pi$ folds all of that lookahead into an improved recommendation.

This gives us a **teacher that is always one step ahead of the student**:

```
   student (raw network)  ──MCTS(lookahead)──►  teacher (search policy π)
        ▲                                                    │
        │                train network toward π and z        │
        └────────────────────────────────────────────────◄──┘
                     student catches up, teacher moves ahead again
```

Because the teacher is built *from* the student but strengthened by search, the
student can always improve by chasing it — no external data, no human games, no
fixed opponent required. This is the formal answer to the "it only plays weak
opponents" worry from [Chapter 5](05-self-play-and-games.md).

## 7.4 The two training targets

Every self-play position produces one training example with two targets:

| Target | Symbol | Where it comes from | Trains |
| --- | --- | --- | --- |
| **Search policy** | $\pi$ | MCTS visit counts at that position | the policy head |
| **Game outcome** | $z$ | the final result of the game, from this position's mover's view: $+1$ win, $0$ draw, $-1$ loss | the value head |

Two things are worth flagging:

- $z$ is a **reward** — ground truth. A win is a win regardless of how the game
  was played. This is the only truly external signal in the whole system.
- $\pi$ is **not** a reward. It's a *distillation of search* — the network is
  taught to imitate what MCTS concluded. It is only as good as the search that
  produced it, which is why more simulations yield better training targets.

In our project, $z$ is computed once per game and stamped onto every stored
position ([`selfplay.py`](../src/chesszero/selfplay.py)), and $\pi$ is captured
per move via `policy_from_visits` ([`mcts.py`](../src/chesszero/mcts.py)) — see
[Chapter 12](12-self-play-in-code.md).

## 7.5 📐 The loss function

Training minimizes a single combined loss over sampled positions:

$$L(\theta) = \underbrace{(z - v)^2}_{\text{value loss}} \;\underbrace{-\; \boldsymbol{\pi}^\top \log \mathbf{p}}_{\text{policy loss}} \;+\; \underbrace{c\,\lVert\theta\rVert^2}_{\text{regularization}}$$

Let's break down each term.

**Value loss $(z - v)^2$** — a mean-squared error. It pushes the value head's
prediction $v$ toward the actual game outcome $z$. If the network thought a
position was winning ($v = 0.8$) but the game was lost ($z = -1$), this term is
large and the gradients correct the network's optimism.

**Policy loss $-\boldsymbol{\pi}^\top \log \mathbf{p}$** — a cross-entropy between
the search policy $\pi$ (target) and the network's policy $\mathbf{p}$ (prediction).
Written out, $-\sum_a \pi(a)\log p(a)$. It is minimized when $\mathbf{p} = \pi$,
i.e. when the network's instant hunch matches what the search concluded. This is
the term that transfers "thinking" into "instinct."

**Regularization $c\lVert\theta\rVert^2$** — a small penalty on the size of the
weights (L2 / weight decay) to prevent overfitting. In our code this is applied
by the optimizer, not written explicitly in the loss.

You'll see this loss implemented almost verbatim in
[Chapter 13](13-training-and-replay-buffer.md):

```python
log_probs   = F.log_softmax(policy_logits, dim=1)
policy_loss = -(policies_t * log_probs).sum(dim=1).mean()   # −πᵀ log p
value_loss  = F.mse_loss(value_pred, values_t)              # (z − v)²
loss = policy_loss + value_loss                             # (+ weight decay in Adam)
```

## 7.6 The loop as generalized policy iteration

Step back and notice the shape of the whole thing. In
[Chapter 3](03-mdps-and-value-functions.md) we met **policy iteration**:
alternate between *evaluating* a policy and *improving* it, and you converge to
the optimal policy. AlphaZero is a scaled-up, approximate version of exactly this
loop — often called **generalized policy iteration**:

```
   ┌─────────────────────────────────────────────────────────────┐
   │                                                               │
   │   POLICY IMPROVEMENT              POLICY EVALUATION            │
   │   ─────────────────               ─────────────────           │
   │   MCTS turns f_θ into a           Self-play games measure      │
   │   stronger policy π and           how good the current         │
   │   picks strong moves      ───►    agent really is (outcomes z) │
   │        ▲                                     │                 │
   │        │                                     ▼                 │
   │        │              TRAIN f_θ toward (π, z)                  │
   │        └─────────────────────◄───────────────                 │
   │             the network now represents a better policy         │
   └─────────────────────────────────────────────────────────────┘
```

- **Improvement:** MCTS + the network produce a better policy $\pi$ than the
  network alone (§7.3).
- **Evaluation:** self-play games, scored by their outcomes $z$, tell us how good
  the current agent is and provide value targets.
- **Representation:** training folds both back into $\theta$, so the network
  *becomes* a better policy and value function.

Each turn of the loop ratchets the agent upward. Combined with the self-play
convergence argument from [Chapter 5](05-self-play-and-games.md), this is why
starting from random weights and pure self-play can reach superhuman strength.

## 7.7 The full algorithm

Here is AlphaZero, end to end, in one box. Every line maps to a module you'll
build in Part III.

```
────────────────────────────────────────────────────────────────────
 ALPHAZERO  (self-play reinforcement learning)
────────────────────────────────────────────────────────────────────
 initialize network fθ with random weights            → network.py
 initialize an empty replay buffer                     → replay_buffer.py

 repeat for each ITERATION:                             → cli.py: cmd_loop

     # ---- PHASE A: self-play (network frozen) ----    → selfplay.py
     repeat for each GAME:
         s ← starting position
         game_history ← []
         while game not over:
             π ← MCTS(s, fθ)          # search improves policy → mcts.py
             record (encode(s), π, side_to_move)        → encoding.py
             a ← sample a move from π (temperature)
             s ← apply a to s
         z ← game outcome (+1 / 0 / −1)
         for each (state, π, side) in game_history:
             value ← z from that side's perspective
             add (state, π, value) to replay buffer

     # ---- PHASE B: training (network updated) ----    → train.py
     repeat several epochs:
         sample a batch (s, π, z) from the buffer
         (p, v) ← fθ(s)
         minimize  (z − v)²  −  πᵀ log p  +  c‖θ‖²
         update θ by gradient descent

     save a checkpoint of fθ                             → checkpoint.py
────────────────────────────────────────────────────────────────────
```

Read it twice: an outer loop of iterations, each with a **self-play phase** that
*uses* the frozen network to generate data, and a **training phase** that
*changes* the network from that data. (This answers Exercise 3 of
[Chapter 1](01-introduction.md).)

## 7.8 How AlphaZero differs from AlphaGo

It's worth knowing the lineage, because it clarifies what's essential.

| | AlphaGo (2016) | AlphaZero (2017) |
| --- | --- | --- |
| Human game data | **Yes** — trained on expert games first | **None** — pure self-play from random |
| Leaf evaluation | value net **+ random rollouts** | value net **only** (no rollouts) |
| Networks | separate policy and value networks | **one** network, two heads |
| Domain knowledge | Go-specific features | just the rules |
| Games it plays | Go only | chess, shogi, and Go with the same code |

The trend is toward *less* built-in knowledge and *more* learned from scratch.
Our project follows AlphaZero: no human games (Pillar 2), no rollouts (the value
head does evaluation), one two-headed network. (The successor, **MuZero**, drops
even the requirement of knowing the rules — it *learns* a model of the game. We
touch on it in [Chapter 18](18-glossary-and-references.md).)

## 7.9 Where every idea lives in the code

You now have the complete theory. Part III builds it for real. Here's your map
from concept to file:

| Concept (this chapter) | Chapter | Module |
| --- | --- | --- |
| Encoding $s$, legal-move masking | [9](09-encoding-board-and-moves.md) | [`encoding.py`](../src/chesszero/encoding.py) |
| The two-headed network $f_\theta$ | [10](10-the-neural-network.md) | [`network.py`](../src/chesszero/network.py) |
| MCTS producing $\pi$ and search value | [11](11-mcts-in-code.md) | [`mcts.py`](../src/chesszero/mcts.py) |
| Self-play generating $(s, \pi, z)$ | [12](12-self-play-in-code.md) | [`selfplay.py`](../src/chesszero/selfplay.py) |
| The loss and replay buffer | [13](13-training-and-replay-buffer.md) | [`train.py`](../src/chesszero/train.py), [`replay_buffer.py`](../src/chesszero/replay_buffer.py) |
| The generalized-policy-iteration loop | [14](14-the-reinforcement-loop.md) | [`cli.py`](../src/chesszero/cli.py) |

That closes Part II. You understand *why* the algorithm works. Turn the page and
we start building it — beginning with the plumbing: project setup and the shape
of the codebase.

---

## Key takeaways

- AlphaZero is a loop: a **network** powers a **search**; the search plays better
  than the network; the network is trained to imitate the search and predict
  outcomes; repeat.
- **MCTS is a policy-improvement operator** — $\pi = \text{MCTS}(s, f_\theta)$ is
  a better policy than the network's raw $\mathbf{p}$. That's the engine of
  improvement.
- Each position yields two targets: the **search policy $\pi$** (trains the policy
  head via cross-entropy) and the **game outcome $z$** (trains the value head via
  MSE). $z$ is ground truth; $\pi$ is distilled search.
- The loss is $L = (z-v)^2 - \pi^\top\log p + c\lVert\theta\rVert^2$.
- The whole thing is **generalized policy iteration**: improve (search), evaluate
  (self-play outcomes), represent (train), repeat.
- Versus AlphaGo: **no human data, no rollouts, one two-headed network** — less
  built-in knowledge, more learned from scratch.

## Exercises

1. Explain in one sentence why $\pi$ (from MCTS) is a *better* policy than
   $\mathbf{p}$ (from the raw network). Which chapter's theorem does this
   instantiate?
2. Which of the two training targets is "ground truth," and which is a
   "distillation of search"? What would happen to learning if you set the MCTS
   simulation count to 1?
3. Match each line of the pseudocode in §7.7 to a module in the table in §7.9.
   Which lines *use* the network and which line *changes* it?
4. AlphaGo evaluated leaves with a value net **plus** random rollouts; AlphaZero
   dropped the rollouts. Given what you learned in
   [Chapter 6](06-monte-carlo-tree-search.md), why were rollouts safe to remove?
5. Write out the loss $L$ and label which term trains the policy head, which
   trains the value head, and which prevents overfitting.

---

> **Course:** [Home](README.md) · **Prev:** [6. Monte Carlo Tree Search](06-monte-carlo-tree-search.md) · **Next:** [8. Project Setup & Architecture](08-project-setup-and-architecture.md)
