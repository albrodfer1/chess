# Chapter 1 — Introduction & The Big Picture

> **Course:** [Home](README.md) · **Prev:** — · **Next:** [2. RL Fundamentals](02-rl-fundamentals.md)

**What you'll learn**
- What Reinforcement Learning is, in one sentence and one picture
- The astonishing idea at the heart of AlphaZero — learning from *nothing but the rules*
- The three properties our chess agent must have, and why each is hard
- A bird's-eye view of the whole system, so later chapters have a place to land

---

## 1.1 A machine that teaches itself

In 2017, a program called **AlphaZero** started knowing only the *rules* of chess
— no opening theory, no grandmaster games, no human advice about what a good
move looks like. It played against itself. A lot. Within hours it was playing
chess better than any human who has ever lived, and better than the strongest
hand-crafted chess engines of the time.

It did this with a loop so simple you can hold it in your head:

```
play games against yourself  →  learn from who won  →  play better games  →  repeat
```

That loop is **Reinforcement Learning**, and by the end of this course you will
have built a working (if smaller) version of it. This is not a toy imitation —
the project in this repository uses the same core algorithm: a neural network
that provides intuition, a tree search that turns intuition into good moves, and
a self-play loop that turns good moves into better intuition.

## 1.2 What is Reinforcement Learning?

Here is the whole field in one sentence:

> **Reinforcement Learning is learning what to do — how to map situations to
> actions — so as to maximize a numerical reward, by trying things and observing
> the consequences.**

Contrast it with the two kinds of learning you may already know:

| Paradigm | You are given… | You learn to… |
| --- | --- | --- |
| **Supervised learning** | examples *with the right answer* (image → "cat") | reproduce the answers |
| **Unsupervised learning** | examples *with no labels* | find structure |
| **Reinforcement learning** | a *reward signal* from an environment you act in | act to earn more reward |

The defining feature of RL is that **nobody tells the agent the right action**.
There is no label saying "the best move here is Nf3." The agent only ever finds
out, sometimes much later, whether things went well (it won) or badly (it lost).
It must figure out *for itself* which of its many earlier decisions deserve the
credit or the blame. That is what makes RL both powerful and hard.

### The canonical picture

Every RL problem is a conversation between two parties — an **agent** and an
**environment** — repeated over and over:

```
            ┌─────────────────── action  a_t ──────────────────┐
            │                                                   ▼
      ┌───────────┐                                      ┌─────────────┐
      │           │                                      │             │
      │   AGENT   │                                      │ ENVIRONMENT │
      │  (learner)│                                      │  (the game) │
      │           │ ◄──── state s_{t+1}, reward r_{t+1} ─┤             │
      └───────────┘                                      └─────────────┘
```

At each step the agent sees the **state** (in chess: the position on the board),
chooses an **action** (a move), and the environment responds with a new state
and a **reward**. In chess the reward is brutally sparse: it's `0` for every
single move of the game, and then at the very end, `+1` if you won, `-1` if you
lost, `0` for a draw. From that whisper of a signal, the agent must learn to play.

We'll make every one of these words precise in [Chapter 2](02-rl-fundamentals.md).
For now, just hold the picture: *act, observe, get reward, improve.*

## 1.3 Why chess is a perfect and a punishing teacher

Chess is an ideal RL laboratory for a few reasons — and a nightmare for a few others.

**Why it's ideal:**
- The rules are known and exact. The environment is a *perfect simulator* — we
  can play as many games as we have compute for, for free.
- It's **two-player and zero-sum**: your win is exactly your opponent's loss.
  This lets the agent generate its own opponents (itself), which is the trick
  that makes self-play work.
- The reward is unambiguous: win, lose, or draw.

**Why it's punishing:**
- The number of possible positions is astronomically large (more than atoms in
  the observable universe). You cannot store a value for each one; you must
  **generalize** with a function approximator — a neural network.
- The reward is **sparse and delayed**. A blunder on move 8 might only cost you
  the game on move 50. Assigning credit across dozens of moves is the central
  difficulty.
- Most sequences of legal moves are terrible. Stumbling onto good play by random
  exploration alone is hopeless — we need *search* to focus effort.

The rest of the course is, in a sense, a set of tools for coping with these
three punishments: neural networks for **generalization**, self-play returns for
**credit assignment**, and Monte Carlo Tree Search for **focused exploration**.

## 1.4 The three pillars of our agent

The project was built to satisfy three requirements. They map directly onto the
three great ideas of AlphaZero, and each gets its own chapter in Part III.

### Pillar 1 — It can never play an illegal move

A chess engine that plays illegal moves is not a chess engine. Our network
outputs a number for *every conceivable* move (4672 of them), but before we ever
act, we **mask** out everything illegal and renormalize over what's left. The
agent literally cannot select an illegal move.

> Built in [`encoding.py`](../src/chesszero/encoding.py) and used in
> [`mcts.py`](../src/chesszero/mcts.py). Covered in [Chapter 9](09-encoding-board-and-moves.md).

### Pillar 2 — It learns by playing itself

There is no database of human games anywhere in this project. A single network
plays *both* sides of every training game. When the game ends, every position is
labelled with the result, and the network is trained to predict it. Because the
opponent is always a copy of yourself, the difficulty is always perfectly
matched to your current skill — an automatic curriculum.

> Built in [`selfplay.py`](../src/chesszero/selfplay.py). Covered in [Chapter 12](12-self-play-in-code.md).

### Pillar 3 — It evaluates positions with Monte Carlo Tree Search

The network's instant opinion of a position is fast but shallow. **MCTS** takes
that opinion and *thinks*: it looks ahead, imagining moves and replies, and
averages the network's evaluations over many simulated lines into a much sharper
judgment — and a better move to play.

> Built in [`mcts.py`](../src/chesszero/mcts.py). Covered in [Chapter 11](11-mcts-in-code.md).

## 1.5 The system at a glance

Here is the entire project on one page. Don't worry about the details — this is a
**map**, so that when we zoom into a module later you'll know where you are.

```
                    ┌──────────────────────────────────────────────┐
                    │              THE REINFORCEMENT LOOP            │
                    │                  (cli.py: cmd_loop)            │
                    └──────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                                                   ▼
   ┌───────────────────────┐                        ┌──────────────────────────┐
   │   PHASE A: SELF-PLAY   │                        │    PHASE B: TRAINING      │
   │     (selfplay.py)      │                        │        (train.py)         │
   │                        │   examples             │                           │
   │  for each move:        │  (state, π, z)         │  minimize:                │
   │    run MCTS ──────────►│ ─────────────────────► │   MSE(value, z)           │
   │      (mcts.py)         │   via ReplayBuffer     │   + CrossEntropy(policy,π)│
   │    pick a move         │                        │                           │
   │  label with result z   │                        │  update the network ──────┼──┐
   └───────────────────────┘                        └──────────────────────────┘  │
              ▲                                                                     │
              │                   better network                                   │
              └─────────────────────────────────────────────────────────────────◄─┘

   Supporting cast:
     encoding.py  — turns a board into numbers the network can read, and masks illegal moves
     network.py   — ChessNet: the neural network with a policy head and a value head
     checkpoint.py— saves/loads the trained network
     agent.py     — uses the trained network to pick moves for real play
     viewer.py    — a browser viewer to watch saved games move by move
```

Read that diagram top to bottom: a loop that alternates between **playing games
with the current network** (Phase A) and **improving the network from those
games** (Phase B). Everything else is machinery in service of those two phases.

## 1.6 How the course unfolds

- **Part I (Chapters 2–4)** builds the vocabulary: states, actions, rewards,
  returns, policies, value functions, and why we approximate them with neural
  networks. No chess yet — just the ideas.
- **Part II (Chapters 5–7)** assembles those ideas into the AlphaZero method:
  self-play, Monte Carlo Tree Search, and the training loop that ties them
  together.
- **Part III (Chapters 8–15)** builds the actual project, one module at a time,
  connecting every line back to the theory from Parts I and II.
- **Part IV (Chapters 16–18)** teaches you to run, debug, and extend it, and
  gives you a glossary and a reading list.

You now have the big picture. In the next chapter we'll slow down and define,
carefully, the handful of terms that the entire field — and the rest of this
course — is built on.

---

## Key takeaways

- **Reinforcement Learning** is learning to act so as to maximize a reward, by
  trial and observation — nobody hands the agent the right answers.
- Every RL problem is an **agent** repeatedly taking **actions** in an
  **environment** that returns new **states** and **rewards**.
- Chess is a perfect simulator but punishes us with a huge state space, sparse
  delayed rewards, and a needle-in-a-haystack space of good moves.
- Our agent stands on three pillars: **legal-move masking**, **self-play**, and
  **Monte Carlo Tree Search** — the same trio behind AlphaZero.
- The whole system is a loop: **self-play** to generate games, **training** to
  improve the network, repeat.

## Exercises

1. In your own words, what is the difference between the label in supervised
   learning and the reward in reinforcement learning? Which one does chess give
   you, and *when*?
2. The reward in chess is `0` for almost every move. Why might that make learning
   harder than, say, a game that gives you points on every turn?
3. Look at the system diagram in §1.5. Which phase *uses* the network, and which
   phase *changes* it? (We'll confirm your answer in [Chapter 14](14-the-reinforcement-loop.md).)
4. Open [`src/chesszero/`](../src/chesszero/) and list the files. Match each one
   to a box in the diagram. Which file do you think is the "brain"?

---

> **Course:** [Home](README.md) · **Prev:** — · **Next:** [2. RL Fundamentals](02-rl-fundamentals.md)
