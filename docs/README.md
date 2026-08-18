# ChessZero — From Zero to Hero

### A hands-on course in Reinforcement Learning, built around a real AlphaZero-style chess engine

Welcome. This course teaches you Reinforcement Learning (RL) *by building a
complete, working chess agent that learns entirely from self-play* — the same
family of ideas behind DeepMind's AlphaZero. We start from "what is a reward?"
and end with a trained network, a Monte Carlo Tree Search, a self-play loop, and
a browser-based game viewer.

Every concept is paired with the **actual code in this repository** (`src/chesszero/`),
so theory never floats free of practice. By the end you will understand not just
*how* the project works, but *why* each design decision was made.

---

## Who this is for

- You can read and write **Python** and have seen a neural network before (you
  know what a layer, a loss, and gradient descent are).
- You do **not** need any prior RL knowledge. We build it from scratch.
- You do **not** need to be a chess player. We use only the rules, and even
  those are handled by a library (`python-chess`).

If you can follow a `for` loop and you're curious how a machine can teach itself
to play a game superhumanly with *no human game data*, you're in the right place.

---

## What you'll build

By the end of the course you'll understand — line by line — an agent with three
defining properties:

1. **It can never play an illegal move** (moves are masked to the legal set).
2. **It learns purely by playing against itself** (no human games, no opening book).
3. **It evaluates positions with Monte Carlo Tree Search** guided by a neural network.

…and you'll be able to run it:

```bash
poetry install
poetry run chesszero loop --iterations 8 --games 10 --sample-games 10   # train
poetry run chesszero viewer                                             # watch it play
```

---

## How to use this course

- **Read in order.** Each chapter builds on the last. Concepts introduced in
  Part I are used without re-explanation in Part III.
- **Keep the code open.** When a chapter cites a file like
  `src/chesszero/mcts.py`, open it. The prose and the code are meant to be read
  together.
- **Do the exercises.** Most chapters end with a few. They range from "predict
  what this line does" to "change a hyperparameter and observe."
- **Run things.** Theory sticks when you watch a loss go down or a game replay.

### Conventions

| Marker | Meaning |
| --- | --- |
| `file.py:42` | a clickable reference to a specific file and line |
| **Key takeaways** | the 3–5 things to remember from a chapter |
| **Exercises** | optional practice to cement understanding |
| 📐 | a math aside — safe to skim on a first read |
| ⚠️ | a common misconception or pitfall |

---

## Table of contents

### Part I — Reinforcement Learning Foundations
*The vocabulary and core ideas of RL, independent of chess.*

1. [Introduction & The Big Picture](01-introduction.md) — what we're building and the intuition behind it
2. [Reinforcement Learning Fundamentals](02-rl-fundamentals.md) — agents, environments, states, actions, rewards, policies, value
3. [MDPs & Value Functions](03-mdps-and-value-functions.md) — the formal model, returns, the Bellman equations, optimality
4. [Deep RL & Function Approximation](04-deep-rl-and-function-approximation.md) — why we need neural networks, policy vs. value methods

### Part II — The AlphaZero Method
*How the ideas combine into a self-improving game player.*

5. [Games, Self-Play & Zero-Sum Search](05-self-play-and-games.md) — minimax, the self-play curriculum, why it converges
6. [Monte Carlo Tree Search](06-monte-carlo-tree-search.md) — from random rollouts to PUCT with a neural guide
7. [The AlphaZero Algorithm](07-the-alphazero-algorithm.md) — policy+value networks, search as policy improvement, the full loop

### Part III — Building ChessZero
*Constructing the real project, module by module.*

8. [Project Setup & Architecture](08-project-setup-and-architecture.md) — Poetry, the `src/` layout, the module map
9. [Encoding the Board & Moves](09-encoding-board-and-moves.md) — 19 planes, the 4672-action space, and the legal mask *(requirement 1)*
10. [The Neural Network](10-the-neural-network.md) — a residual tower with policy and value heads
11. [Monte Carlo Tree Search in Code](11-mcts-in-code.md) — `Evaluator`, `Node`, PUCT selection, back-ups *(requirement 3)*
12. [Self-Play](12-self-play-in-code.md) — generating games and labelling training data *(requirement 2)*
13. [Training & the Replay Buffer](13-training-and-replay-buffer.md) — the loss function and how examples flow
14. [The Reinforcement Loop & Checkpoints](14-the-reinforcement-loop.md) — tying self-play and training together
15. [Playing, Evaluating & the Game Viewer](15-playing-evaluating-viewer.md) — the agent, `eval`, and the graphical interface

### Part IV — Going Further
*Understanding, debugging, and scaling what you've built.*

16. [Debugging & Understanding Training](16-debugging-and-convergence.md) — reading the numbers, failure modes, what to watch
17. [Scaling Up & Improvements](17-scaling-and-improvements.md) — bigger nets, tree reuse, resignation, distributed self-play
18. [Glossary & Further Reading](18-glossary-and-references.md) — every term in one place, plus the seminal papers

---

## The map: course ↔ code

```
Part I–II  (theory)          Part III  (this repo)
─────────────────────        ────────────────────────────────
reward, return, value   ──►  selfplay.py, train.py, mcts.py
policy & value network  ──►  network.py  (ChessNet)
state / action encoding ──►  encoding.py (19 planes, 4672 actions)
MCTS / PUCT             ──►  mcts.py     (run_mcts, _select_child)
self-play curriculum    ──►  selfplay.py (play_game)
policy iteration loop   ──►  cli.py      (cmd_loop)
```

Ready? Start with **[Chapter 1 — Introduction & The Big Picture](01-introduction.md)**.
