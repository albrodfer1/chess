# Chapter 2 — Reinforcement Learning Fundamentals

> **Course:** [Home](README.md) · **Prev:** [1. Introduction](01-introduction.md) · **Next:** [3. MDPs & Value Functions](03-mdps-and-value-functions.md)

**What you'll learn**
- The precise vocabulary of RL: state, action, reward, return, policy, value
- Why we care about *cumulative* reward (the return) and what the discount factor γ does
- What an episode is, and why one chess game is exactly one episode
- The RL objective — the single quantity everything tries to maximize
- A first, intuitive look at value functions, and the crucial difference between *reward* and *value*
- The exploration–exploitation dilemma, and model-free vs. model-based learning

---

## 2.1 The interaction loop, made precise

In [Chapter 1](01-introduction.md) we drew a picture of an **agent** talking to an
**environment**. Let's make it exact, because every symbol here will reappear
throughout the course.

Time proceeds in discrete steps $t = 0, 1, 2, \dots$. At each step:

1. The agent observes the **state** $s_t$.
2. The agent picks an **action** $a_t$.
3. The environment returns a **reward** $r_{t+1}$ and the next **state** $s_{t+1}$.

```
   s_0 ──a_0──► (r_1, s_1) ──a_1──► (r_2, s_2) ──a_2──► ... ──► (r_T, s_T)
   │                                                                  │
   start                                                         terminal
```

That sequence of states, actions, and rewards is called a **trajectory**:

$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$$

In our chess project, one full trajectory is produced by
[`selfplay.play_game`](../src/chesszero/selfplay.py) — the `while` loop that pushes
moves onto a board until the game ends is *literally* this loop.

## 2.2 States and observations

The **state** $s_t$ is the complete situation the agent finds itself in. In
chess, the state is the position: where every piece stands, whose turn it is, and
a few extra facts we'll meet in [Chapter 3](03-mdps-and-value-functions.md).

You'll sometimes hear the word **observation** instead of *state*. The distinction:

- A **state** contains everything relevant about the world.
- An **observation** is whatever the agent actually gets to see, which might be
  incomplete (a poker player can't see opponents' cards).

Chess is **fully observable**: both players see the whole board, so observation =
state. This is a gift — it means the agent's input can, in principle, capture
everything it needs. In code, the state is turned into numbers by
[`encoding.encode_board`](../src/chesszero/encoding.py), which produces a
`(19, 8, 8)` tensor. We'll dissect those 19 planes in the next chapter; for now,
just know: **board position → array of numbers = the state the network sees.**

## 2.3 Actions and the action space

An **action** $a_t$ is a choice the agent makes. In chess, an action is a move:
"knight from g1 to f3."

The **action space** $\mathcal{A}$ is the set of all actions the agent could ever
take. This is a subtle and important design decision. In our project the action
space is **fixed at 4672 possible moves** — every conceivable (from-square,
to-square, promotion) combination on an 8×8 board:

```python
# encoding.py
NUM_PLANES = 73            # 56 "queen" moves + 8 knight moves + 9 underpromotions
ACTION_SIZE = NUM_PLANES * 64  # 4672  (73 move-types × 64 from-squares)
```

⚠️ **Most of those 4672 actions are illegal in any given position.** A pawn can't
teleport; you can't move a piece that isn't there. This is exactly Pillar 1 from
Chapter 1: the agent must choose only from the *legal* subset. RL theory calls
the legal actions in state $s$ the set $\mathcal{A}(s) \subseteq \mathcal{A}$. Our
code enforces this with [`encoding.legal_mask`](../src/chesszero/encoding.py),
which we cover in [Chapter 9](09-encoding-board-and-moves.md).

## 2.4 Reward: the scalar that defines the goal

The **reward** $r_t$ is a single number the environment hands back after each
action. It is the *only* signal that tells the agent what it's supposed to
achieve. This is the famous **reward hypothesis**:

> Everything we mean by "goals" and "purposes" can be captured as maximizing the
> expected cumulative value of a single scalar reward.

In chess, the reward is stark. Almost every move earns reward `0`. Only when the
game *ends* does a non-zero reward appear:

| Game outcome | Reward |
| --- | --- |
| You win | $+1$ |
| Draw | $0$ |
| You lose | $-1$ |

Look at where this lives in the code — [`selfplay._game_result`](../src/chesszero/selfplay.py):

```python
def _game_result(board):
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0                                   # draw / unfinished
    return 1.0 if outcome.winner == chess.WHITE else -1.0
```

This is a **sparse** and **delayed** reward: you get a whisper of feedback (one
number) only at the very end, and it must somehow justify every one of the ~40
moves that led there. Handling that delay is the whole game — and RL's central
trick for it is the **value function** (§2.7).

> 📐 **Reward design.** In some RL problems you *shape* rewards to give hints
> ("+0.1 for capturing a piece"). AlphaZero deliberately does **not**: the only
> reward is win/draw/loss. This avoids baking in human misconceptions about what
> "good chess" looks like, and lets the agent discover strategy from scratch.

## 2.5 The return: why we sum rewards

The agent doesn't care about the immediate reward alone — it cares about *all the
reward it will collect from now on*. That cumulative future reward is the
**return**, written $G_t$:

$$G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^{k}\, r_{t+k+1}$$

The new symbol $\gamma$ (gamma) is the **discount factor**, a number in $[0, 1]$.
It says how much a reward in the future is worth *right now*:

- $\gamma = 0$: totally myopic — only the next reward matters.
- $\gamma = 1$: totally far-sighted — a reward 100 steps away counts just as much
  as one now.
- $\gamma = 0.99$: a common middle ground — future rewards matter, but nearer ones
  matter more.

**Why discount at all?** Three reasons:

1. **Math**: in never-ending tasks the sum could be infinite; $\gamma < 1$ keeps
   it finite.
2. **Uncertainty**: the far future is harder to predict, so weight it less.
3. **Preference**: often we genuinely prefer reward sooner than later.

**In chess, $\gamma \approx 1$.** Because there is exactly one non-zero reward and
it comes at the end, the return of *every* position in a game is essentially just
the final result. If you won, the return of every position you saw is $+1$; if you
lost, $-1$. This is precisely what the self-play code computes when it labels each
stored position with the game's outcome:

```python
# selfplay.play_game — after the game ends:
for state, policy, side_to_move in history:
    value = result if side_to_move == chess.WHITE else -result   # G_t for this state
    examples.append(Example(state=state, policy=policy, value=float(value)))
```

Notice the sign flip: the return is always measured **from the perspective of the
player to move** at that state. If White eventually won (`result = +1`), then a
position where it was Black's turn has a return of $-1$ — bad news for Black. Keep
this "whose turn is it" bookkeeping in mind; it returns in [Chapter 5](05-self-play-and-games.md)
and [Chapter 11](11-mcts-in-code.md).

## 2.6 Episodes: one game, one episode

A task that always reaches an end is called **episodic**, and each run from start
to finish is an **episode**. Chess is episodic: every game ends (someone is
checkmated, or it's a draw, or — in our code — a move limit is hit). The terminal
step $T$ is when `board.is_game_over()` becomes true.

```python
# selfplay.play_game
while not board.is_game_over() and move_number < config.max_moves:
    ...   # one iteration = one time step t
```

So: **one chess game = one episode = one trajectory $\tau$.** During training we
play thousands of these episodes, and each one contributes a batch of
`(state, policy, return)` examples to learn from. (Contrast with *continuing*
tasks, like balancing a pole forever, which never terminate — those rely more
heavily on discounting.)

## 2.7 The policy: how the agent decides

The agent's behaviour is captured by its **policy**, written $\pi$. A policy is a
rule for choosing actions from states. It comes in two flavours:

- **Deterministic**: $a = \pi(s)$ — the same state always yields the same action.
- **Stochastic**: $\pi(a \mid s)$ — a *probability distribution* over actions;
  the agent samples from it.

RL usually works with stochastic policies, because randomness is how the agent
*explores* (§2.8). In our project the policy is stochastic, and it is produced in
two stages:

```
board ──► ChessNet ──► a probability for each move   (fast "intuition")
                │
                ▼
              MCTS ──► refined probabilities from visit counts  (slow "thinking")
                │
                ▼
         sample or pick the move to play
```

- The neural network [`ChessNet`](../src/chesszero/network.py) outputs a raw
  policy — a probability for each of the (legal) moves. This is the agent's
  instant hunch.
- **Monte Carlo Tree Search** ([`mcts.py`](../src/chesszero/mcts.py)) improves
  that hunch by looking ahead, producing a sharper policy from how often it
  visited each move during search.
- Finally [`select_move`](../src/chesszero/mcts.py) turns those numbers into an
  actual choice.

We will build each stage in Part III. The key idea for now: **the policy is the
thing we are ultimately trying to improve.** A better policy = better play.

### The RL objective

With policy and return defined, we can finally state what "learning" means in one
line. The agent seeks the policy that maximizes the **expected return** from the
start:

$$\pi^{*} = \arg\max_{\pi}\; \mathbb{E}_{\tau \sim \pi}\big[\, G_0 \,\big]$$

In plain words: *find the way of playing that, on average over many games, wins
the most.* Everything else in this course — networks, search, self-play — is
machinery for climbing toward that $\pi^{*}$.

## 2.8 Value functions: a first look (and reward ≠ value)

If the return $G_t$ is what we want to maximize, it would be enormously useful to
*predict* it before the game ends. That prediction is the **value function**.

- **State-value** $V^{\pi}(s)$: "If I'm in state $s$ and act according to policy
  $\pi$ from here on, what return should I expect?"
- **Action-value** $Q^{\pi}(s, a)$: "If I'm in state $s$, take action $a$ *now*,
  and then follow $\pi$, what return should I expect?"

Intuitively, $V$ answers *how good is this position?* and $Q$ answers *how good is
this move?* We define them formally in [Chapter 3](03-mdps-and-value-functions.md).

⚠️ **Reward is not value — do not confuse them.** This trips up almost everyone:

| | Reward $r_t$ | Value $V(s)$ |
| --- | --- | --- |
| What it is | immediate feedback for one step | *expected total future* reward |
| Who provides it | the environment (a fact) | the agent's *estimate* (a guess) |
| In chess | 0 every move, ±1 at the end | a prediction, e.g. "+0.7: I'm probably winning" |

A quiet position early in the game has **reward 0** — nothing just happened — but
might have **high value** because it's likely to *lead* to a win. The value head
of [`ChessNet`](../src/chesszero/network.py) outputs exactly this kind of estimate:
a number in $[-1, +1]$ predicting the eventual result. It is trained to match the
return $G_t$ (the game outcome), which is the reward summed over the episode.

## 2.9 Exploration vs. exploitation

Here is a dilemma with no perfect answer. At any moment the agent can:

- **Exploit**: play the move it currently believes is best.
- **Explore**: try something else, to *discover* whether an alternative is
  actually better.

Pure exploitation is a trap. If the agent always plays what looks best *right
now*, it will keep playing the same handful of openings and never discover the
brilliant line it hasn't tried. But pure exploration never puts its knowledge to
use. Good learning needs a balance, tilted toward exploration early (when the
agent knows little) and toward exploitation later (when its judgment is trustworthy).

Our project injects exploration in two concrete ways, both of which you'll meet
in [Chapter 12](12-self-play-in-code.md):

- **Dirichlet noise** added to the policy at the root of every search
  (`run_mcts(..., add_noise=True)`), which randomly boosts some moves so even
  unpromising-looking ones occasionally get tried.
- **Temperature** sampling early in the game:

```python
# selfplay.play_game
temperature = 1.0 if move_number < config.temperature_moves else 0.0
move = select_move(root, temperature=temperature)
```

With `temperature = 1.0` the agent *samples* moves in proportion to how good they
seem (exploration); with `temperature = 0.0` it always takes the single best move
(exploitation). Early moves explore to diversify games; later moves exploit to
play the game out well.

## 2.10 Model-free vs. model-based

One last fork in the road. Does the agent have access to a **model** of the
environment — a way to predict "if I take action $a$ in state $s$, what state and
reward come next"?

- **Model-free** RL learns purely from experience, without ever simulating ahead.
  It's like learning to drive only by driving.
- **Model-based** RL has (or learns) a model and can *plan* by imagining
  consequences before acting. It's like rehearsing a route in your head.

**Chess is a gift for model-based methods: we have a perfect model.** The rules
are known exactly, so from any position we can enumerate every legal move and see
precisely which position results — for free, without touching the real game. The
`python-chess` library *is* that model.

This is the foundation of Pillar 3. Because we can simulate ahead perfectly,
**Monte Carlo Tree Search** ([Chapter 6](06-monte-carlo-tree-search.md)) can plan
by rolling forward through imagined moves, using the network's value estimates as
a stand-in for "how this line turns out." Model-free intuition (the network) plus
model-based planning (the search) is the combination that makes AlphaZero tick.

---

## Key takeaways

- The RL loop is: observe **state** $s_t$ → take **action** $a_t$ → receive
  **reward** $r_{t+1}$ and next state $s_{t+1}$. One chess game is one **episode**.
- The **return** $G_t = \sum_k \gamma^k r_{t+k+1}$ is cumulative future reward;
  the **discount** $\gamma$ trades off near vs. far. In chess $\gamma \approx 1$
  and the return of every position is just the final result.
- A **policy** $\pi(a\mid s)$ maps states to action probabilities; the RL
  objective is to find the policy with the highest **expected return**.
- **Value** predicts future return and is *not* the same as reward: a position can
  have zero reward yet high value.
- The agent must balance **exploration** and **exploitation** (our code uses
  Dirichlet noise + temperature), and because chess gives us a **perfect model**,
  we can plan ahead with search (model-based) on top of a learned network
  (model-free).

## Exercises

1. Write out the trajectory $\tau$ for a 3-move game that ends in White's victory.
   What is the reward at each step? What is the return $G_t$ of each state
   (take $\gamma = 1$)?
2. A position is completely quiet — no capture just happened. What is its
   *reward*? Could its *value* still be large? Explain the difference in one
   sentence.
3. In [`selfplay.play_game`](../src/chesszero/selfplay.py), the line
   `temperature = 1.0 if move_number < config.temperature_moves else 0.0`
   controls exploration. What behaviour would you get if you hard-coded
   `temperature = 0.0` for the whole game? Why might that hurt *training*?
4. Chess is "model-based" because we have a perfect simulator of the rules. Name a
   real-world problem where you would *not* have such a model, forcing you to be
   model-free.
5. The action space is 4672 but only ~30 moves are legal in a typical position.
   Why is it still convenient to give the network a fixed-size output of 4672
   rather than "just the legal moves"? (Hint: think about what a neural network's
   output layer looks like.)

---

> **Course:** [Home](README.md) · **Prev:** [1. Introduction](01-introduction.md) · **Next:** [3. MDPs & Value Functions](03-mdps-and-value-functions.md)
