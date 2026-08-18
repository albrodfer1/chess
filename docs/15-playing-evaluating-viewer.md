# Chapter 15 — Playing, Evaluating & the Game Viewer

> **Course:** [Home](README.md) · **Prev:** [14. The Reinforcement Loop & Checkpoints](14-the-reinforcement-loop.md) · **Next:** [16. Debugging & Understanding Training](16-debugging-and-convergence.md)

**What you'll learn**
- How a trained network is turned into a move-picking agent — and why it *cannot* play an illegal move
- Playing against the model yourself from the terminal
- Measuring real strength with `eval` (model vs. model, or model vs. random)
- The dependency-free browser **game viewer**: how it serves games and what the evaluation bars mean
- The exact commands to produce games and launch the viewer

---

## 15.1 From a network to a player: `MCTSAgent`

Training (Chapters [12](12-self-play-in-code.md)–[14](14-the-reinforcement-loop.md))
gives us a checkpoint. To *use* it — to actually play chess — we need something
that takes a board and returns a move. That is `MCTSAgent` in
[`agent.py`](../src/chesszero/agent.py):

```python
class MCTSAgent:
    """Chooses moves by running MCTS and picking the most-visited root child.

    Because MCTS only ever expands legal moves, this agent can never return an
    illegal move (requirement 1).
    """

    def __init__(self, net, config, simulations=None):
        self.config = config
        if simulations is not None:
            self.config = Config(**{**config.__dict__})
            self.config.num_simulations = simulations
        self.evaluator = Evaluator(net, self.config.device)

    def choose_move(self, board, temperature=0.0):
        net = self.evaluator.net
        was_training = net.training
        net.eval()
        try:
            root = run_mcts(board, self.evaluator, self.config, add_noise=False)
        finally:
            net.train(was_training)
        return select_move(root, temperature=temperature)
```

Everything here is machinery you already met in [Chapter 11](11-mcts-in-code.md);
`MCTSAgent` just packages it for real play. Four details matter:

1. **Optional simulation override.** If you pass `simulations=`, the agent copies
   the `Config` and bumps `num_simulations` — letting a stronger (slower) or
   weaker (faster) search be requested without disturbing the shared config.
2. **`add_noise=False`.** Self-play injects Dirichlet noise at the root for
   *exploration* ([Chapter 5](05-self-play-and-games.md)); when actually playing,
   we want the agent's honest best move, so noise is off.
3. **`net.eval()` around the search.** Batch-norm must use running statistics, not
   batch statistics, during inference. The `try/finally` restores the previous
   mode so calling `choose_move` never has surprising side effects.
4. **`select_move(root, temperature=0.0)`.** With `temperature=0.0`,
   `select_move` picks the **most-visited** root child — the greedy, strongest
   choice (see [Chapter 11](11-mcts-in-code.md)).

### 15.1.1 Why it *cannot* play an illegal move — requirement 1, again

This is worth stating plainly because it is one of the project's three pillars
([Chapter 1](01-introduction.md), [Chapter 9](09-encoding-board-and-moves.md)).
`choose_move` only ever returns a move that `select_move` pulled from
`root.children`. Those children were created by MCTS expansion, which only ever
adds **legal** moves (the `Evaluator` masks the policy to legal moves before
anything else — [Chapter 11](11-mcts-in-code.md)). So the set of moves the agent
can possibly return is, by construction, a subset of the legal moves. There is no
code path by which an illegal move reaches the board.

```
network logits (4672)
   │  Evaluator.evaluate: keep only LEGAL indices, softmax
   ▼
priors over legal moves ──► MCTS expands only legal children
   │
   ▼
root.children  = { legal move : node }  ──► select_move picks one of these
                                             └─► always legal ✔
```

## 15.2 Playing against the model: `cmd_play`

`cmd_play` in [`cli.py`](../src/chesszero/cli.py) lets you sit across the board
from a checkpoint in your terminal:

```python
def cmd_play(args):
    net, config, _ = load_checkpoint(args.checkpoint, device=args.device or None)
    if args.simulations:
        config.num_simulations = args.simulations
    net.eval()
    agent = MCTSAgent(net, config)

    board = chess.Board()
    human_is_white = args.color == "white"
    ...
    while not board.is_game_over():
        print(board, "\n")
        if board.turn == (chess.WHITE if human_is_white else chess.BLACK):
            move = _read_human_move(board)
            if move is None:
                return
        else:
            print("Thinking...", flush=True)
            move = agent.choose_move(board, temperature=0.0)
            print(f"Model plays: {board.san(move)}\n")
        board.push(move)

    print(board, "\n")
    print("Result:", board.result(), "-", board.outcome())
```

We [load a checkpoint](14-the-reinforcement-loop.md), wrap it in an `MCTSAgent`,
and alternate turns. On the model's turn it "thinks" (runs MCTS) and plays
greedily. On your turn, input is parsed forgivingly:

```python
def _read_human_move(board):
    while True:
        raw = input("Your move: ").strip()
        if raw.lower() in ("quit", "exit"):
            return None
        move = None
        try:
            move = board.parse_san(raw)          # e.g. "Nf3", "e4", "O-O"
        except ValueError:
            try:
                move = chess.Move.from_uci(raw)  # e.g. "g1f3", "e2e4"
            except ValueError:
                move = None
        if move is not None and move in board.legal_moves:
            return move
        print("Illegal or unparseable move; try again.")
```

It accepts either **SAN** (`Nf3`, standard algebraic notation, what humans read in
books) or **UCI** (`g1f3`, the from-square/to-square form engines use), and it
loops until you enter something both parseable *and* legal. The legality check
here protects the *human* from cheating or fat-fingering — the model's legality is
already guaranteed by §15.1.1.

Try it (after training a checkpoint):

```bash
poetry run chesszero play checkpoints/latest.pt --color white
poetry run chesszero play checkpoints/latest.pt --color black --simulations 400
```

More `--simulations` makes the model think harder (and move slower).

## 15.3 Measuring strength: `cmd_eval`

Playing a few games yourself is fun but anecdotal. To *measure* whether training
is working, we pit models against each other, or against a random player as a
sanity-check baseline. That is `cmd_eval`:

```python
def cmd_eval(args):
    net_a, config_a, _ = load_checkpoint(args.model_a, device=args.device or None)
    net_a.eval()
    agent_a = MCTSAgent(net_a, config_a, simulations=args.simulations or None)

    agent_b = None
    if args.model_b:
        net_b, config_b, _ = load_checkpoint(args.model_b, device=args.device or None)
        net_b.eval()
        agent_b = MCTSAgent(net_b, config_b, simulations=args.simulations or None)

    wins_a = wins_b = draws = 0
    for i in range(args.games):
        a_is_white = i % 2 == 0
        result = _play_match(agent_a, agent_b, a_is_white, config_a.max_moves)
        if result == 0:
            draws += 1
        elif (result == 1) == a_is_white:
            wins_a += 1
        else:
            wins_b += 1
        print(f"game {i + 1}/{args.games}: A={wins_a} B={wins_b} draws={draws}", flush=True)

    label_b = "model_b" if args.model_b else "random"
    print(f"\nmodel_a wins: {wins_a} | {label_b} wins: {wins_b} | draws: {draws}")
```

Two important design choices:

- **`agent_b` is `None` ⇒ opponent is random.** If you don't pass `--model-b`, the
  opponent plays uniformly random legal moves. Beating a random player convincingly
  is the *floor* every trained model should clear — a great first sanity check
  ([Chapter 16](16-debugging-and-convergence.md)).
- **Colors alternate** (`a_is_white = i % 2 == 0`). White has a small first-move
  advantage, so swapping colors every game keeps the comparison fair. The line
  `(result == 1) == a_is_white` decodes the white-perspective result into "did A
  win?": `_play_match` returns `+1` if White won, and A was White exactly when
  `a_is_white` is true.

The match itself:

```python
def _play_match(agent_a, agent_b, a_is_white, max_moves):
    import random
    board = chess.Board()
    moves = 0
    while not board.is_game_over() and moves < max_moves:
        a_turn = board.turn == (chess.WHITE if a_is_white else chess.BLACK)
        if a_turn:
            move = agent_a.choose_move(board, temperature=0.0)
        elif agent_b is not None:
            move = agent_b.choose_move(board, temperature=0.0)
        else:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        moves += 1

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0
```

Both agents play greedily (`temperature=0.0`), the game is capped at `max_moves`
(a cap reached with no winner counts as a draw, `0.0`), and the result is reported
from White's perspective.

Typical uses:

```bash
# Sanity check: is the latest model clearly better than random?
poetry run chesszero eval checkpoints/latest.pt --games 20

# Progress check: does iteration 8 beat iteration 1?
poetry run chesszero eval checkpoints/model_iter_8.pt --model-b checkpoints/model_iter_1.pt --games 20
```

> 📐 **This is why we kept numbered checkpoints.** In
> [Chapter 14](14-the-reinforcement-loop.md) we saved `model_iter_N.pt` every
> iteration. `eval` with `--model-b` is the payoff: it lets you confirm that later
> networks actually beat earlier ones — the empirical signature of a working RL
> loop. A newer network that *cannot* beat an older one is a red flag
> ([Chapter 16](16-debugging-and-convergence.md)).

## 15.4 The game viewer

Numbers tell you *whether* the agent improved; the **viewer** lets you *see* how it
plays and how confident it is about each move. It replays the games sampled during
training ([Chapter 14](14-the-reinforcement-loop.md), §14.5) and, for every ply,
shows the network's move evaluations.

### 15.4.1 What was saved

Recall from [Chapter 12](12-self-play-in-code.md) that sampled games are stored as
JSON with a per-ply record built by `_record_ply`:

```python
return {
    "ply": ply,
    "fen": board.fen(),
    "turn": "white" if board.turn == chess.WHITE else "black",
    "played": {"uci": move.uci(), "san": board.san(move)},
    "value": round(net_value, 5),        # network value, side-to-move view
    "mcts_value": round(root.value, 5),  # MCTS search value, side-to-move view
    "evaluations": evaluations[:12],     # top 12 by network probability
    "terminal": False,
}
```

Each entry in `evaluations` pairs a move's **network policy softmax** (`policy`)
with its **MCTS visit fraction** (`visits`) — the two quantities the viewer draws
as bars. Importantly, `policy` here is a *clean* re-evaluation, free of the
Dirichlet noise that self-play adds ([Chapter 12](12-self-play-in-code.md)), so
what you see is the network's honest opinion.

### 15.4.2 The server: `viewer.py`

The viewer needs no web framework and no internet. `run_viewer` in
[`viewer.py`](../src/chesszero/viewer.py) starts a tiny local HTTP server:

```python
def run_viewer(games_dir="games", port=8000, open_browser=True):
    games_dir = Path(games_dir).resolve()
    if not games_dir.exists():
        raise SystemExit("Games directory not found ... Run training with --sample-games first")
    html_bytes = VIEWER_HTML.read_bytes()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(games_dir), **kwargs)
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(html_bytes, "text/html; charset=utf-8")
            elif self.path == "/manifest.json":
                body = json.dumps(_build_manifest(games_dir)).encode()
                self._send(body, "application/json")
            else:
                super().do_GET()
    ...
```

It serves three kinds of thing:

| Request | Response |
| --- | --- |
| `/` or `/index.html` | the packaged `viewer/index.html` GUI (read once at startup) |
| `/manifest.json` | a freshly built list of the games in `games_dir` |
| anything else | the matching file from `games_dir` (i.e. the game JSONs) |

`_build_manifest` scans the directory for game files and summarizes each one so the
GUI can populate its dropdown:

```python
def _build_manifest(games_dir):
    games = []
    for path in sorted(games_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text())
        games.append({
            "file": path.name,
            "game_index": data.get("game_index"),
            "iteration": data.get("iteration"),
            "result": data.get("result_str", data.get("result")),
            "termination": data.get("termination"),
            "num_plies": data.get("num_plies", len(data.get("moves", []))),
        })
    games.sort(key=lambda g: (g["game_index"] if g["game_index"] is not None else 0))
    return games
```

Because the manifest is rebuilt on every request, you can leave the viewer running
during training and just refresh the page to see newly saved games appear.

### 15.4.3 The GUI: `viewer/index.html`

The front end is a single, dependency-free HTML file (vanilla JavaScript — no CDN,
works offline). It:

- **Renders the board from FEN** using Unicode chess glyphs (♔♕♖♗♘♙ / ♚♛♜♝♞♟),
  drawing all 64 squares in light/dark colors — no image assets needed.
- **Steps through moves** with ⏮ ◀ ▶ ⏭ buttons, a slider, the **← / →** arrow
  keys, and **spacebar** to toggle autoplay.
- **Highlights the played move** by tinting its from- and to-squares.
- **Draws the evaluation bars** for each candidate move, and marks the move that
  was actually played with a ✓.
- **Shows the value gauge** — the position value from the side-to-move's
  perspective, both the network's estimate and the MCTS-refined value.

```
┌─────────────────────────┐   Move 24 (white to move)
│  8  ♜ · · · ♚ · · ♜      │   played: Nf3 ✓
│  7  ♟ ♟ ♟ · · ♟ ♟ ♟      │   value  +0.31 (net)  +0.44 (mcts)
│  ...                     │
│  1  ♖ · · ♕ ♔ · · ♖      │   evaluations (policy ▮  visits ▮):
└─────────────────────────┘     Nf3 ✓ ▮▮▮▮▮▮▮▮▮▮  52%   ▮▮▮▮▮▮ 61%
   [⏮] [◀] [▶] [⏭]  ▶ play      e4    ▮▮▮▮        21%   ▮▮▮▮   28%
   ├────────●──────────┤        Bc4   ▮▮          9%    ▮      6%
```

Reading the two bars together is the whole point:

- **Blue = policy (network softmax).** The network's *instant intuition*: how much
  probability it placed on each move before searching ([Chapter 10](10-the-neural-network.md)).
- **Green = visits (MCTS).** How the *search* redistributed its effort after
  looking ahead ([Chapter 11](11-mcts-in-code.md)). This is the improved policy π
  that becomes the training target ([Chapter 13](13-training-and-replay-buffer.md)).

When the green bars concentrate more sharply than the blue ones, you are literally
watching search *sharpen* the network's raw guess — policy improvement, made
visible ([Chapter 7](07-the-alphazero-algorithm.md)).

### 15.4.4 Running the viewer

First produce some games (if you haven't already), then launch:

```bash
# 1. Train and sample games spread across the run (Chapter 14)
poetry run chesszero loop --iterations 8 --games 10 --sample-games 10

# 2. Launch the viewer (opens your browser at http://127.0.0.1:8000/)
poetry run chesszero viewer

# Options:
poetry run chesszero viewer --games-dir games   # where the JSONs live
poetry run chesszero viewer --port 9000          # use a different port
poetry run chesszero viewer --no-browser         # just serve; open the URL yourself
```

## 15.5 Watching the agent learn

Here is the satisfying part. Because games were sampled *evenly across the run*
([Chapter 14](14-the-reinforcement-loop.md)), the dropdown holds an early game, a
late game, and several in between. Flip between the first and last:

- In **early** games the policy bars are flat and spread out — the network has no
  idea which moves are good — and games are long, meandering, and often decided by
  a random-looking blunder.
- In **late** games the policy mass **concentrates** on a few sensible moves, the
  value gauge tracks the eventual result more faithfully, and the play is visibly
  more purposeful.

That visual shift — from flat to peaked evaluation bars — is the same thing the
falling `policy_loss` in [Chapter 14](14-the-reinforcement-loop.md)'s logs is
telling you, but you can *see* it move by move. We turn to reading those numbers,
and diagnosing when they *don't* improve, in
[Chapter 16](16-debugging-and-convergence.md).

---

## Key takeaways

- `MCTSAgent.choose_move` wraps a trained network in MCTS with **no root noise**
  and **greedy** move selection; by construction it can only return legal moves
  (requirement 1).
- `cmd_play` lets you play against a checkpoint, accepting **SAN or UCI** input and
  rejecting anything illegal or unparseable.
- `cmd_eval` measures strength by playing many games — against another checkpoint
  or a **random baseline** — with **alternating colors** for fairness. This is how
  you confirm later iterations beat earlier ones.
- The **viewer** is a dependency-free local web app: `viewer.py` serves the
  packaged HTML plus a live manifest and the game JSONs; the GUI renders the board
  from FEN and draws **policy (network)** vs. **visits (MCTS)** bars per move.
- Comparing an early and a late sampled game makes learning *visible*: evaluation
  mass concentrates on stronger moves over time.

## Exercises

1. In §15.1.1, walk the chain of code that makes an illegal move from the agent
   impossible. Which single function is the "gatekeeper," and what would break the
   guarantee if you edited it?
2. In `cmd_eval`, explain the line `(result == 1) == a_is_white`. Construct a case
   where A is Black and wins, and confirm the bookkeeping counts it as an A win.
3. Why does `MCTSAgent` set `add_noise=False` for real play but self-play sets it
   `True`? What would happen to `eval` results if you left noise on? (See
   [Chapter 5](05-self-play-and-games.md).)
4. Train a tiny run with `--sample-games 6`, launch the viewer, and compare the
   earliest and latest games. Describe how the **policy** (blue) and **visits**
   (green) bars differ between them.
5. The viewer rebuilds the manifest on every request. What convenience does that
   buy you if you keep the viewer open in a browser tab while a `loop` is still
   running?

---

> **Course:** [Home](README.md) · **Prev:** [14. The Reinforcement Loop & Checkpoints](14-the-reinforcement-loop.md) · **Next:** [16. Debugging & Understanding Training](16-debugging-and-convergence.md)
