# Chapter 18 — Glossary & Further Reading

> **Course:** [Home](README.md) · **Prev:** [17. Scaling Up & Improvements](17-scaling-and-improvements.md) · **Next:** —

**What you'll learn**
- A one-stop definition of every term used in the course, cross-linked to where it first appears
- The seminal papers and references, with a one-line note on what each contributed
- A recap of the whole course, and concrete next steps to keep going

---

## 18.1 How to use this glossary

Terms are grouped by theme (RL foundations, games & search, deep learning, this project's specifics) and, within each group, ordered roughly the way the course introduces them. Each entry is one or two lines, with a link to the chapter that develops it. Skim it now to see the shape of the field; return to it whenever a term feels fuzzy.

## 18.2 Reinforcement learning foundations

- **Agent** — the learner and decision-maker; the thing that chooses moves. In this project, the network + MCTS together. ([Ch. 1](01-introduction.md), [Ch. 2](02-rl-fundamentals.md))
- **Environment** — everything the agent acts on and gets feedback from; here, the chess board and rules (simulated by `python-chess`). ([Ch. 2](02-rl-fundamentals.md))
- **State ($s$)** — a complete description of the situation at one moment; a chess position (plus castling rights, en passant, clock). ([Ch. 2](02-rl-fundamentals.md), [Ch. 3](03-mdps-and-value-functions.md))
- **Observation** — what the agent actually perceives of the state. In fully-observed chess, observation = state. ([Ch. 2](02-rl-fundamentals.md))
- **Action ($a$)** — a choice the agent makes; a legal chess move. ([Ch. 2](02-rl-fundamentals.md))
- **Action space** — the set of all possible actions. Here it's the fixed AlphaZero encoding of **4672** move slots, most illegal in any given position. ([Ch. 9](09-encoding-board-and-moves.md))
- **Reward ($r$)** — the scalar feedback signal from the environment. In chess it is `0` every move and then `+1 / 0 / −1` at the end (win/draw/loss). ([Ch. 1](01-introduction.md), [Ch. 2](02-rl-fundamentals.md))
- **Return ($G_t$)** — the cumulative (optionally discounted) reward from time $t$ onward; what the agent actually tries to maximize. ([Ch. 2](02-rl-fundamentals.md))
- **Discount factor ($\gamma$)** — weights future rewards vs. immediate ones, $0 \le \gamma \le 1$. In chess the reward is terminal, so $\gamma \approx 1$. ([Ch. 2](02-rl-fundamentals.md))
- **Episode / trajectory** — one complete run from start to terminal state; here, one full game. ([Ch. 2](02-rl-fundamentals.md))
- **Policy ($\pi$)** — the agent's strategy: a mapping from states to a distribution over actions, $\pi(a\mid s)$. ([Ch. 2](02-rl-fundamentals.md))
- **Value function ($V^\pi(s)$)** — the expected return from state $s$ when following policy $\pi$; "how good is this position?" ([Ch. 3](03-mdps-and-value-functions.md))
- **Action-value function ($Q^\pi(s,a)$)** — the expected return from taking action $a$ in state $s$, then following $\pi$; "how good is this move?" ([Ch. 3](03-mdps-and-value-functions.md))
- **Exploration vs. exploitation** — the tension between trying new actions to gather information and taking the current best-known action. Handled here by Dirichlet noise and temperature. ([Ch. 2](02-rl-fundamentals.md), [Ch. 6](06-monte-carlo-tree-search.md))
- **Credit assignment** — the problem of deciding which earlier decisions deserve credit/blame for a delayed outcome. Chess's central difficulty. ([Ch. 1](01-introduction.md), [Ch. 12](12-self-play-in-code.md))
- **Model-free vs. model-based** — whether the agent has a model of the environment's dynamics. Chess is *model-based*: we have a perfect simulator (the rules), which is exactly what makes MCTS lookahead possible. ([Ch. 2](02-rl-fundamentals.md))

## 18.3 MDPs, values, and dynamic programming

- **Markov property** — the future depends only on the present state, not the path taken to reach it. ([Ch. 3](03-mdps-and-value-functions.md))
- **Markov Decision Process (MDP)** — the formal RL model: states $S$, actions $A$, transition function $P$, reward function $R$, discount $\gamma$. ([Ch. 3](03-mdps-and-value-functions.md))
- **Transition function ($P$)** — the probability of the next state given the current state and action. In chess, deterministic given a move. ([Ch. 3](03-mdps-and-value-functions.md))
- **Bellman equation** — a recursive relationship expressing a state's value in terms of its successors' values; the backbone of value-based RL. ([Ch. 3](03-mdps-and-value-functions.md))
- **Optimality ($V^*$, $Q^*$)** — the value functions of the best possible policy. ([Ch. 3](03-mdps-and-value-functions.md))
- **Policy iteration / policy improvement** — alternating between *evaluating* a policy and *improving* it toward the greedy policy w.r.t. its values; provably climbs to the optimum. **MCTS is a policy-improvement operator**, which is why AlphaZero's loop works. ([Ch. 3](03-mdps-and-value-functions.md), [Ch. 7](07-the-alphazero-algorithm.md))
- **Generalized policy iteration** — the general pattern of interleaving approximate evaluation and improvement; the frame for the whole self-play loop. ([Ch. 7](07-the-alphazero-algorithm.md))

## 18.4 Function approximation & deep learning

- **Function approximation** — using a parameterized function (e.g. a neural net) to represent a value or policy when the state space is too large to tabulate. Essential for chess's $10^{40}$+ states. ([Ch. 4](04-deep-rl-and-function-approximation.md))
- **Generalization** — the approximator's ability to give sensible outputs on states it never saw, by exploiting similarity to states it did. ([Ch. 4](04-deep-rl-and-function-approximation.md))
- **Deep RL** — reinforcement learning with deep neural networks as the function approximators. ([Ch. 4](04-deep-rl-and-function-approximation.md))
- **Value-based methods** — learn a value/Q-function and act greedily w.r.t. it (e.g. DQN). ([Ch. 4](04-deep-rl-and-function-approximation.md))
- **Policy-gradient methods** — adjust the policy's parameters directly in the direction of higher return. ([Ch. 4](04-deep-rl-and-function-approximation.md))
- **Actor-critic** — a hybrid: an *actor* (policy) guided by a *critic* (value). AlphaZero resembles this but trains toward **search** targets rather than classic policy gradients. ([Ch. 4](04-deep-rl-and-function-approximation.md), [Ch. 7](07-the-alphazero-algorithm.md))
- **Residual block** — a network building block with a skip connection (`out = relu(F(x) + x)`), letting deep towers train stably. Our tower stacks `num_res_blocks` of these. ([Ch. 10](10-the-neural-network.md))
- **Batch normalization** — normalizes layer activations across a batch to stabilize and speed up training; used in every conv block of `ChessNet`. ([Ch. 10](10-the-neural-network.md))
- **Policy head** — the network output branch producing a distribution (as logits) over the 4672 actions. ([Ch. 10](10-the-neural-network.md))
- **Value head** — the network output branch producing a single scalar in $[-1, 1]$ (via `tanh`) estimating the outcome from the side-to-move's view. ([Ch. 10](10-the-neural-network.md))
- **Logits** — raw, unnormalized network outputs, *before* softmax. The policy head emits logits, not probabilities. ([Ch. 10](10-the-neural-network.md), [Ch. 11](11-mcts-in-code.md))
- **Softmax** — turns logits into a probability distribution. Applied over **legal moves only** in `Evaluator.evaluate`. ([Ch. 9](09-encoding-board-and-moves.md), [Ch. 11](11-mcts-in-code.md))
- **Cross-entropy** — the loss comparing two distributions; used to fit the policy head to the MCTS policy π. ([Ch. 13](13-training-and-replay-buffer.md))
- **Mean squared error (MSE)** — the regression loss; used to fit the value head to the game outcome z. ([Ch. 13](13-training-and-replay-buffer.md))
- **Adam** — the adaptive gradient-descent optimizer used to update the network. ([Ch. 13](13-training-and-replay-buffer.md))
- **Weight decay / L2 regularization** — a penalty on large weights (Adam's `weight_decay`) that improves generalization; the $c\lVert\theta\rVert^2$ term of the AlphaZero loss. ([Ch. 7](07-the-alphazero-algorithm.md), [Ch. 13](13-training-and-replay-buffer.md))
- **Minibatch** — a small random subset of training examples used for one gradient step; drawn from the replay buffer via `sample`. ([Ch. 13](13-training-and-replay-buffer.md))
- **Replay buffer** — a fixed-size store of recent experience (`deque(maxlen=…)`) that decorrelates samples and mitigates forgetting. ([Ch. 13](13-training-and-replay-buffer.md), [Ch. 16](16-debugging-and-convergence.md))

## 18.5 Games, search, and self-play

- **Zero-sum game** — one player's gain is exactly the other's loss; chess. Enables self-play. ([Ch. 5](05-self-play-and-games.md))
- **Minimax** — the principle that each side plays to maximize its own outcome, which minimizes the opponent's; encoded here by the **sign flip** on value back-ups. ([Ch. 5](05-self-play-and-games.md), [Ch. 11](11-mcts-in-code.md))
- **Game tree** — the branching tree of positions reachable by sequences of moves. ([Ch. 5](05-self-play-and-games.md), [Ch. 6](06-monte-carlo-tree-search.md))
- **Self-play** — the agent generating training games by playing against a copy of itself. ([Ch. 5](05-self-play-and-games.md), [Ch. 12](12-self-play-in-code.md))
- **Curriculum** — a sequence of tasks of increasing difficulty. Self-play is an *automatic* curriculum: the opponent is always exactly your strength. ([Ch. 5](05-self-play-and-games.md))
- **Monte Carlo Tree Search (MCTS)** — a search that spends a fixed budget of simulations, focusing on promising lines, to turn a cheap evaluator into strong play. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 11](11-mcts-in-code.md))
- **Simulation** — one MCTS iteration: select a leaf, evaluate it, back the value up. `num_simulations` of these run per move. ([Ch. 6](06-monte-carlo-tree-search.md))
- **Rollout** — in classic MCTS, a random play-out to the game's end to estimate a leaf's value. **AlphaZero replaces rollouts with the value head** (bootstrapping). ([Ch. 6](06-monte-carlo-tree-search.md))
- **Bootstrapping** — estimating a value from another learned estimate (the value head at a non-terminal leaf) rather than from a full play-out. Why low simulation counts still yield a value. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 16](16-debugging-and-convergence.md))
- **UCT (Upper Confidence bounds applied to Trees)** — the classic selection rule balancing a node's average value with an exploration bonus. ([Ch. 6](06-monte-carlo-tree-search.md))
- **PUCT** — AlphaZero's variant of UCT that folds in the network's **prior**: score $= Q + c_{puct}\,P\,\frac{\sqrt{\sum N}}{1 + N}$. Implemented in `_select_child`. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 11](11-mcts-in-code.md))
- **Prior ($P$)** — the network policy's probability for a move, used to bias which branches MCTS explores first; stored on each `Node`. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 11](11-mcts-in-code.md))
- **Visit count ($N$)** — how many simulations passed through a node. The distribution of root-child visits *is* the improved policy π. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 11](11-mcts-in-code.md))
- **Backup (backpropagation)** — propagating a leaf's value up the search path, updating each node's `value_sum` and `visit_count`, flipping sign each ply. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 11](11-mcts-in-code.md))
- **Dirichlet noise** — random noise mixed into the **root** priors during self-play (`_add_dirichlet_noise`) to force exploration; controlled by `dirichlet_alpha` and `dirichlet_epsilon`. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 12](12-self-play-in-code.md))
- **Temperature** — a knob on move selection: high temperature samples moves proportional to visit counts (exploratory); temperature 0 always picks the most-visited move (greedy). Applied for the first `temperature_moves` plies. ([Ch. 6](06-monte-carlo-tree-search.md), [Ch. 12](12-self-play-in-code.md))

## 18.6 This project's specifics

- **`ChessNet`** — the residual policy-value network in [`network.py`](../src/chesszero/network.py). ([Ch. 10](10-the-neural-network.md))
- **`Evaluator`** — wraps `ChessNet` to score one board, masking illegal moves and returning `(priors, value)`. ([Ch. 11](11-mcts-in-code.md))
- **`Node`** — a node in the MCTS tree (`prior`, `visit_count`, `value_sum`, `children`, lazy `board`). ([Ch. 11](11-mcts-in-code.md))
- **`Example`** — one training datum: `(state, policy, value)` = `(19×8×8 tensor, 4672-vector π, scalar z)`. ([Ch. 13](13-training-and-replay-buffer.md))
- **Legal mask** — the boolean vector marking legal-move indices; the mechanism guaranteeing the agent never selects an illegal move (**requirement 1**). ([Ch. 9](09-encoding-board-and-moves.md))
- **Plane** — one 8×8 channel of the board encoding; the encoding uses **19** planes (pieces, side-to-move, castling, en passant, clock). ([Ch. 9](09-encoding-board-and-moves.md))
- **Ply** — a single move by one player (a "half-move"). Two plies make one full move. Game length is measured in plies here. ([Ch. 12](12-self-play-in-code.md))
- **FEN (Forsyth–Edwards Notation)** — a compact text string encoding a full chess position; used in saved game records and rendered by the viewer. ([Ch. 12](12-self-play-in-code.md), [Ch. 15](15-playing-evaluating-viewer.md))
- **SAN (Standard Algebraic Notation)** — human-readable move notation (`Nf3`, `exd5`); accepted by `cmd_play` and shown in the viewer. ([Ch. 15](15-playing-evaluating-viewer.md))
- **UCI (Universal Chess Interface) move** — a move written as from-square + to-square (`e2e4`); also accepted by `cmd_play`. ([Ch. 15](15-playing-evaluating-viewer.md))
- **Promotion / underpromotion** — a pawn reaching the last rank becoming another piece; queen-promotions are folded into "queen moves" and knight/bishop/rook underpromotions get dedicated planes in the action encoding. ([Ch. 9](09-encoding-board-and-moves.md))
- **Checkpoint (`.pt` file)** — a saved snapshot (model weights, optimizer state, config, iteration) written by `save_checkpoint`; the `.pt` extension is PyTorch convention. ([Ch. 14](14-the-reinforcement-loop.md))
- **Iteration** — one full turn of the loop: a batch of self-play games followed by a training phase and a checkpoint. ([Ch. 14](14-the-reinforcement-loop.md))

## 18.7 Further reading

**The foundational textbook**
- **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)** — the canonical, freely-available RL text. Read it for everything in Parts I–II: MDPs, value functions, Bellman equations, temporal-difference learning, and policy methods. If you read one thing after this course, read this.

**The papers that built this line of work**
- **Silver et al., "Mastering the game of Go with deep neural networks and tree search" (AlphaGo, 2016)** — first to beat a top human at Go, combining supervised learning from human games, self-play RL, and MCTS with rollouts.
- **Silver et al., "Mastering the game of Go without human knowledge" (AlphaGo Zero, 2017)** — dropped human games entirely, unified policy and value into one network, and replaced rollouts with the value head. Introduced the exact self-play + MCTS + policy-iteration loop this project implements.
- **Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero, 2018)** — generalized AlphaGo Zero to chess and shogi with essentially no game-specific changes. This is the algorithm you built.
- **Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero, 2020)** — removed even the requirement of knowing the rules, by *learning* a model of the environment's dynamics alongside the policy and value. The natural next step beyond AlphaZero.

**The tools**
- **python-chess documentation** — the library providing board state, legal move generation, FEN/SAN/UCI parsing, and game outcomes. Everything in [`encoding.py`](../src/chesszero/encoding.py) and [`selfplay.py`](../src/chesszero/selfplay.py) rests on it.
- **PyTorch documentation** — the deep-learning framework behind [`network.py`](../src/chesszero/network.py) and [`train.py`](../src/chesszero/train.py): tensors, autograd, `nn.Module`, optimizers, and device management.

## 18.8 Course recap

You started from a single sentence — *learn to act so as to maximize reward* — and ended with a working, self-improving chess engine. Along the way you learned to:

- speak the language of RL: **states, actions, rewards, returns, policies, and values** (Part I);
- formalize it as an **MDP** and see why huge state spaces force **function approximation** with neural networks (Part I);
- combine **self-play**, **MCTS**, and a **policy-value network** into the AlphaZero loop, and understand *why* it converges as **generalized policy iteration** (Part II);
- read the project line by line — the **19-plane encoding** and **legal mask** (requirement 1), the **residual network**, **PUCT search** (requirement 3), **self-play** (requirement 2), the **training loss**, and the **reinforcement loop** (Part III);
- and run, debug, and extend it — reading the metrics, diagnosing failure modes, and knowing which knobs to turn (Part IV).

The three requirements you set out to satisfy are now concrete code you understand: the agent **cannot play illegal moves**, it **learns only from itself**, and it **evaluates positions with Monte Carlo Tree Search**.

## 18.9 Where to go next

Pick one and build:

1. **Implement an improvement from [Chapter 17](17-scaling-and-improvements.md).** The easiest high-value ones: persist the replay buffer across `--resume`, add resignation to `play_game`, or add arena gating using the existing `_play_match`.
2. **Train longer, on a GPU.** Scale `num_res_blocks`, `num_filters`, `num_simulations`, and `iterations`, and use `cmd_eval` to chart real strength gains iteration over iteration.
3. **Port the algorithm to a simpler game.** Connect Four or Tic-Tac-Toe have tiny state and action spaces, so a strong agent trains in *minutes* — perfect for experimenting with the loop, MCTS, and encoding without waiting hours. The only pieces you'd rewrite are the encoding and the environment; the MCTS, network, training, and loop transfer almost unchanged.
4. **Read the papers in §18.7** with the code open. Now that you've implemented the algorithm, the AlphaGo Zero and AlphaZero papers read like annotated versions of what you built.

You reached the end. You went from zero to a working AlphaZero. Now go make it stronger.

---

> **Course:** [Home](README.md) · **Prev:** [17. Scaling Up & Improvements](17-scaling-and-improvements.md) · **Next:** —
