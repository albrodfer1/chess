"""Encoding between python-chess objects and neural-network tensors.

This implements the AlphaZero action representation: an 8x8x73 = 4672 flat
action space, plus a 21-plane board encoding. It also exposes the legal-move
mask that guarantees the agent can never *select* an illegal move.
"""

from __future__ import annotations

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------
# Plane layout (all 8x8):
#   0-5   : white pieces  (P, N, B, R, Q, K)
#   6-11  : black pieces  (P, N, B, R, Q, K)
#   12    : side to move  (all ones if white to move)
#   13-16 : castling rights (W kingside, W queenside, B kingside, B queenside)
#   17    : en-passant target square
#   18    : halfmove clock, normalized by 100
#   19    : repetition — current position has occurred at least twice
#   20    : repetition — current position has occurred at least three times
#
# The two repetition planes give the network the history it needs to see a
# threefold-repetition draw coming: without them a single-position encoding is
# blind to how many times the position has already appeared in the game.
INPUT_PLANES = 21

# Action space: 73 move "planes" per from-square, 64 from-squares.
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

QUEEN_PLANES = 56          # 8 directions x 7 distances
KNIGHT_PLANES = 8
UNDERPROMOTION_PLANES = 9  # 3 pieces x 3 file-directions
NUM_PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMOTION_PLANES  # 73
ACTION_SIZE = NUM_PLANES * 64  # 4672


def _piece_plane(color: bool, piece_type: int) -> int:
    """Return the input plane index for a (color, piece_type)."""
    return (piece_type - 1) + (0 if color == chess.WHITE else 6)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board into a (21, 8, 8) float32 tensor (absolute coordinates)."""
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        planes[_piece_plane(piece.color, piece.piece_type), rank, file] = 1.0

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

    # Repetition counting needs the move stack; boards built by pushing moves
    # (self-play and MCTS copies, which keep their history) carry it, so
    # is_repetition is meaningful here.
    if board.is_repetition(2):
        planes[19, :, :] = 1.0
    if board.is_repetition(3):
        planes[20, :, :] = 1.0
    return planes


# ---------------------------------------------------------------------------
# Move <-> index
# ---------------------------------------------------------------------------
def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move to its index in the flat 4672 action space.

    Index = plane * 64 + from_square, so it aligns with a (73, 8, 8)
    convolutional policy head flattened in row-major order.
    """
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


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Reconstruct a chess.Move from a flat action index and the board.

    The board is needed to decide whether a queen-plane pawn move to the back
    rank is a queen promotion.
    """
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
