# Chapter 9 — Encoding the Board & Moves

> **Course:** [Home](README.md) · **Prev:** [8. Project Setup & Architecture](08-project-setup-and-architecture.md) · **Next:** [10. The Neural Network](10-the-neural-network.md)

**What you'll learn**
- Why a chess position must be turned into numbers, and how `encode_board` does it
- Every one of the **19 input planes**, and why castling/en-passant/clock are there
- The AlphaZero **4672-move action space** — 56 + 8 + 9 planes per square — and how `move_to_index` works
- How the **legal mask** guarantees the agent can never select an illegal move *(requirement 1)*
- How this all connects to the Markov property from [Chapter 3](03-mdps-and-value-functions.md)

---

A neural network eats numbers and emits numbers. But our environment speaks
chess: `python-chess` gives us `Board` objects and `Move` objects. This chapter
is the **translator** — [`encoding.py`](../src/chesszero/encoding.py) — that
converts a position into a tensor the network can read, and converts the
network's numeric output back into concrete moves. Crucially, it's also where
**requirement 1** ("no illegal moves") lives.

Everything here is small, pure-function code: no neural network, no search, no
randomness. That makes it the perfect place to be *exact*, because a bug in the
encoding would silently poison everything downstream.

## 9.1 Why encode at all?

Recall the RL picture from [Chapter 2](02-rl-fundamentals.md): the agent observes
a **state** and chooses an **action**. Here:

- the **state** is a chess position (a `chess.Board`), and
- the **action** is a move (a `chess.Move`).

A convolutional neural network ([Chapter 10](10-the-neural-network.md)) expects a
fixed-shape tensor of floats, not a `Board`. And its policy head emits a
fixed-length vector of numbers, not a `Move`. So we need two bridges:

```
   chess.Board  ──encode_board──►  (19, 8, 8) float tensor      (state  → network input)
   chess.Move   ──move_to_index─►  integer in [0, 4672)         (action ↔ network output)
   integer      ──index_to_move─►  chess.Move
```

Let's build each bridge.

## 9.2 Encoding the board: 19 planes

The board is encoded as a stack of **19 "planes,"** each an 8×8 grid — think of
it like a 19-channel image of the chessboard. The header comment in
[`encoding.py`](../src/chesszero/encoding.py) documents the layout exactly:

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

Here is what each group means:

| Planes | Contents | Encoding |
| --- | --- | --- |
| 0–5 | white Pawn, kNight, Bishop, Rook, Queen, King | a `1.0` on each square holding that piece |
| 6–11 | black P, N, B, R, Q, K | same, for black |
| 12 | side to move | **whole plane** is `1.0` if it's White's turn, else `0.0` |
| 13–16 | castling rights | whole plane `1.0`/`0.0` per right (W-kingside, W-queenside, B-kingside, B-queenside) |
| 17 | en-passant target | a single `1.0` on the ep-capture square, if any |
| 18 | halfmove clock | whole plane set to `clock / 100` |

The core loop that fills the piece planes:

```python
def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board into a (19, 8, 8) float32 tensor (absolute coordinates)."""
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        planes[_piece_plane(piece.color, piece.piece_type), rank, file] = 1.0
    ...
```

`board.piece_map()` gives `{square: piece}` for every occupied square. Each
square index (0–63) splits into a `rank` (row, 0–7) and `file` (column, 0–7) via
`divmod(square, 8)`. The helper picks the right plane:

```python
def _piece_plane(color: bool, piece_type: int) -> int:
    return (piece_type - 1) + (0 if color == chess.WHITE else 6)
```

`python-chess` numbers piece types `PAWN=1 … KING=6`, so `piece_type - 1` gives
0–5, and black pieces are shifted by 6 into planes 6–11.

The remaining planes are scalar facts broadcast across the whole 8×8 grid (or,
for en-passant, a single square):

```python
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        planes[17, rank, file] = 1.0

    planes[18, :, :] = board.halfmove_clock / 100.0
    return planes
```

### 📐 Why these extra planes? The Markov property.

In [Chapter 3](03-mdps-and-value-functions.md) we said chess is a **Markov
Decision Process**: the future depends only on the present *state*, not on how
you got there. But is the board *picture* alone enough to be Markovian?

No! Two positions with identical piece placements can be genuinely different
games:

- One side may still have **castling rights**; the other may have lost them.
- An **en-passant** capture may be legally available *this move only*.
- The **halfmove clock** ticks toward the 50-move draw; a position at clock 3 is
  not the same as the identical picture at clock 99.

If we left these out, the state would be *non-Markovian* — the "same" input could
require different optimal play. Planes 13–18 restore the Markov property by
packing exactly the hidden state that the rules of chess care about. This is a
direct, concrete payoff of the theory: **the encoding is designed so that
`encode_board(board)` is a sufficient statistic for optimal play.**

(The one thing deliberately *not* encoded is move-repetition history for the
threefold rule. That's a pragmatic simplification — full AlphaZero stacks several
past positions. We rely instead on automatic draw rules and a move cap, discussed
in [Chapter 12](12-self-play-in-code.md).)

### ⚠️ Absolute coordinates, not "my perspective"

A subtle design choice: this project encodes the board in **absolute
coordinates** — a1 is always the bottom-left — and simply *tells* the network
whose turn it is via plane 12. The docstring flags it: `(absolute coordinates)`.

Full AlphaZero instead flips the board so the side to move always "plays up the
board," which bakes in a useful symmetry (white-to-move and the mirrored
black-to-move look identical to the network). Why not do that here?

- Flipping the board means you must **also** flip every move when mapping the
  policy — an easy place to introduce subtle, hard-to-find bugs.
- The absolute scheme is simpler and obviously correct: a move's encoding never
  depends on whose turn it is.

The tradeoff is that the network must learn the black/white symmetry itself
rather than getting it for free — a little less sample-efficient, but far less
error-prone. For a teaching project, correctness wins.

## 9.3 The action space: 4672 moves

Now the harder bridge: representing *moves* as fixed integer indices, so the
policy head can be a plain vector of 4672 numbers — one per possible move.

AlphaZero's clever encoding is **8 × 8 × 73 = 4672**: for each of the 64
"from" squares, there are 73 possible "move types." The 73 break down as:

```python
QUEEN_PLANES = 56          # 8 directions x 7 distances
KNIGHT_PLANES = 8
UNDERPROMOTION_PLANES = 9  # 3 pieces x 3 file-directions
NUM_PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMOTION_PLANES  # 73
ACTION_SIZE = NUM_PLANES * 64  # 4672
```

| Planes | Move type | Count |
| --- | --- | --- |
| 0–55 | **"Queen" moves**: slide in one of 8 compass directions, 1–7 squares | 8 × 7 = 56 |
| 56–63 | **Knight** moves: the 8 L-shapes | 8 |
| 64–72 | **Underpromotions**: promote to N/B/R while moving straight, or capturing left/right | 3 × 3 = 9 |

Two things worth noting:

- The "queen moves" cover *all* sliding pieces (queen, rook, bishop) and also
  king steps and normal pawn pushes — anything that moves in a straight or
  diagonal line. The name just refers to the *geometry*.
- **Queen promotions** are not in the 9 underpromotion planes. A pawn pushing to
  the last rank is encoded as an ordinary 1-square "queen move," and the promotion
  piece defaults to a queen. Only *under*promotions (to knight/bishop/rook) need
  the special planes, because they're the only case the geometry can't
  distinguish. (You'll see this handled in `index_to_move` below.)

The direction tables that define the geometry:

```python
QUEEN_DIRECTIONS = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]
KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
```

### `move_to_index` — a move to its slot

```python
def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    from_rank, from_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)
    d_rank = to_rank - from_rank
    d_file = to_file - from_file

    promotion = move.promotion

    if promotion is not None and promotion != chess.QUEEN:
        # Underpromotion: pawn advances one rank; file delta in {-1, 0, 1}.
        piece_idx = UNDERPROMOTION_PIECES.index(promotion)
        plane = QUEEN_PLANES + KNIGHT_PLANES + piece_idx * 3 + (d_file + 1)
    elif (abs(d_rank), abs(d_file)) in ((2, 1), (1, 2)):
        plane = QUEEN_PLANES + KNIGHT_MOVES.index((d_rank, d_file))
    else:
        direction = (_sign(d_rank), _sign(d_file))
        distance = max(abs(d_rank), abs(d_file))
        dir_idx = QUEEN_DIRECTIONS.index(direction)
        plane = dir_idx * 7 + (distance - 1)

    return plane * 64 + from_sq
```

Read it as a three-way decision on the *shape* of the move:

1. **Underpromotion?** (promotion set, and not to a queen) → use planes 64–72,
   indexed by which piece and the file shift (`d_file + 1` maps `-1,0,+1` → `0,1,2`).
2. **Knight jump?** (the displacement is an L) → use planes 56–63, indexed by
   which of the 8 L-shapes.
3. **Otherwise a sliding/step move** → find its compass `direction` (via `_sign`
   of each delta) and its `distance` (1–7), giving a plane in 0–55.

The final line packs it: `plane * 64 + from_sq`. The docstring explains the
ordering choice:

> Index = plane * 64 + from_square, so it aligns with a (73, 8, 8) convolutional
> policy head flattened in row-major order.

In other words, the integer index lines up naturally with a convolutional policy
head's output — a small but tidy piece of engineering.

### A worked example

Take White's opening move **e2 → e4** (from a fresh board, no promotion):

- `from_sq = e2 = 12`, so `from_rank, from_file = 1, 4`.
- `to_sq = e4 = 28`, so `to_rank, to_file = 3, 4`.
- `d_rank = 2`, `d_file = 0`.
- Not a promotion, not an L-shape.
- `direction = (sign(2), sign(0)) = (1, 0)` → that's **N**, `dir_idx = 0`.
- `distance = max(2, 0) = 2` → plane `= 0*7 + (2-1) = 1`.
- Index `= 1 * 64 + 12 = 76`.

So the network's output slot **76** means "advance the piece on e2 two squares
north." Every legal (and illegal) move has exactly one such slot.

### `index_to_move` — a slot back to a move

The inverse reconstructs a `chess.Move` from an index. It needs the `board` for
one reason — to decide whether a plain forward pawn move to the back rank should
become a queen promotion:

```python
def index_to_move(index: int, board: chess.Board) -> chess.Move:
    plane, from_sq = divmod(index, 64)
    from_rank, from_file = divmod(from_sq, 8)
    promotion = None

    if plane < QUEEN_PLANES:
        dir_idx, dist = divmod(plane, 7)
        d_rank, d_file = QUEEN_DIRECTIONS[dir_idx]
        d_rank *= dist + 1
        d_file *= dist + 1
    elif plane < QUEEN_PLANES + KNIGHT_PLANES:
        d_rank, d_file = KNIGHT_MOVES[plane - QUEEN_PLANES]
    else:
        under = plane - QUEEN_PLANES - KNIGHT_PLANES
        piece_idx, dir_idx = divmod(under, 3)
        d_file = dir_idx - 1
        d_rank = 1 if from_rank == 6 else -1  # white promotes upward, black down
        promotion = UNDERPROMOTION_PIECES[piece_idx]

    to_rank = from_rank + d_rank
    to_file = from_file + d_file
    to_sq = to_rank * 8 + to_file

    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)
```

This mirrors `move_to_index` exactly, plane-range by plane-range. The final
`if promotion is None` block is the queen-promotion special case: a pawn that
lands on rank 0 or 7 via a queen-plane move is completed as a queen promotion.

## 9.4 The legal mask — requirement 1

Here is the heart of the chapter, and of the first pillar from
[Chapter 1](01-introduction.md): **the agent can never select an illegal move.**

The network outputs a number for all 4672 slots — including thousands of moves
that are illegal in the current position (a knight can't teleport across the
board; a pinned piece can't move). Left unchecked, the agent might "want" to play
one. The fix is a **mask**:

```python
def legal_moves_and_indices(board: chess.Board):
    """Return (list_of_moves, list_of_indices) for all legal moves."""
    moves = list(board.legal_moves)
    indices = [move_to_index(m) for m in moves]
    return moves, indices


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean (4672,) mask that is True exactly on legal-move indices.

    This is the guarantee behind requirement (1): illegal actions are masked
    out before any policy is normalized or any move is sampled.
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
```

The trick is that the set of legal moves comes straight from `python-chess`'s
`board.legal_moves` — a correct, exhaustive generator. We never trust the network
to *know* what's legal; we trust the rules engine, and only let the network
*rank* the moves that the engine has already certified as legal.

### How the mask is actually used

The masking happens for real in `Evaluator.evaluate`
([`mcts.py`](../src/chesszero/mcts.py), covered in [Chapter 11](11-mcts-in-code.md)).
Rather than build a boolean mask, it takes the even more direct route of only
*reading* the logits at legal indices:

```python
    moves = list(board.legal_moves)
    if not moves:
        return {}, float(value.item())

    indices = np.fromiter((move_to_index(m) for m in moves), dtype=np.int64)
    move_logits = logits[indices]
    # Softmax over legal moves only -> illegal moves get zero probability.
    move_logits -= move_logits.max()
    priors = np.exp(move_logits)
    priors /= priors.sum()

    return {m: float(p) for m, p in zip(moves, priors)}, float(value.item())
```

Read that carefully — it's the whole guarantee in five lines:

1. Ask `python-chess` for the legal moves.
2. Gather *only* those slots' logits (`logits[indices]`).
3. Softmax over that legal subset, so the probabilities sum to 1 **over legal
   moves alone**.
4. Return a dict keyed by actual `chess.Move` objects.

Because the probability distribution is *built* from the legal set, an illegal
move has no slot in it at all — probability effectively zero, never selectable.
And since MCTS only ever expands children that appear in this dict
([Chapter 11](11-mcts-in-code.md)), and move selection only ever chooses among
those children ([Chapter 12](12-self-play-in-code.md)), the illegality can't
sneak back in downstream.

### ⚠️ Raw logits are not a move distribution

A common beginner mistake is to `softmax` the network's full 4672-vector and pick
the argmax. That can — and early in training *will* — pick an illegal move,
because an untrained network assigns mass everywhere. The rule to internalize:

> **The policy head's raw output is not a usable move distribution.** It only
> becomes one after masking to the legal set and renormalizing.

## 9.5 How we know it's correct: the tests

Encoding bugs are silent and catastrophic, so [`tests/test_encoding.py`](../tests/test_encoding.py)
pins the behaviour down. The most important test round-trips **every legal move**
from several positions through `move_to_index` and back through `index_to_move`:

```python
def test_move_index_roundtrip_all_legal():
    """Every legal move from a few positions must round-trip through the index."""
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/P7/8/8/8/8/7p/k6K w - - 0 1",  # promotions available
    ]
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            idx = move_to_index(move)
            assert 0 <= idx < ACTION_SIZE
            back = index_to_move(idx, board)
            assert back == move, f"{move} -> {idx} -> {back}"
```

Other tests check the invariants we relied on above:

- `test_action_size` — the action space is exactly `4672`.
- `test_encode_shape_and_side_to_move` — `encode_board` returns `(19, 8, 8)`, and
  plane 12 is all-ones from the starting position (White to move).
- `test_legal_mask_matches_legal_moves` — the mask's `True` count equals the
  number of legal moves, and each legal move's index is set.
- `test_indices_are_unique_per_position` — no two legal moves collide on the same
  index (a collision would let one move "shadow" another).

Run them with `poetry run pytest tests/test_encoding.py`.

## 9.6 Where we go next

We can now turn any position into a `(19, 8, 8)` tensor and any move into an index
in `[0, 4672)` — and we can guarantee the agent only ever considers legal moves.
The next chapter builds the thing that consumes that tensor and produces those
4672 numbers plus a value: the neural network, `ChessNet`.

---

## Key takeaways

- `encode_board` turns a position into a **`(19, 8, 8)` tensor**: 12 piece planes,
  plus side-to-move, castling, en-passant, and the halfmove clock.
- Those extra planes exist to keep the state **Markovian** — the tensor is a
  sufficient statistic for optimal play, exactly as [Chapter 3](03-mdps-and-value-functions.md) requires.
- Moves live in the **4672-slot AlphaZero action space** (56 queen + 8 knight + 9
  underpromotion planes × 64 squares); `move_to_index`/`index_to_move` convert both
  ways.
- The **legal mask** (and the softmax-over-legal-moves in `Evaluator`) is the
  concrete guarantee behind **requirement 1**: illegal moves get zero probability
  and are never selectable.
- Raw policy logits are *not* a move distribution until masked and renormalized.

## Exercises

1. Work out the action index for the knight move **g1 → f3** from the starting
   position. (Hint: `g1 = 6`, `f3 = 21`; find the L-shape in `KNIGHT_MOVES`.)
2. Why must `index_to_move` take the `board` as an argument, while `move_to_index`
   does not? What single case forces the asymmetry?
3. Plane 18 stores `halfmove_clock / 100`. What would go subtly wrong if we
   *omitted* this plane? Frame your answer in terms of the Markov property from
   [Chapter 3](03-mdps-and-value-functions.md).
4. In `Evaluator.evaluate`, the code does `move_logits -= move_logits.max()`
   before exponentiating. This changes nothing mathematically — why is it there?
   (Hint: floating-point overflow in `exp`.)
5. Suppose you deleted the masking and just did `softmax` over all 4672 logits,
   then `argmax`. Describe a concrete situation early in training where the agent
   would attempt an illegal move.

---

> **Course:** [Home](README.md) · **Prev:** [8. Project Setup & Architecture](08-project-setup-and-architecture.md) · **Next:** [10. The Neural Network](10-the-neural-network.md)
