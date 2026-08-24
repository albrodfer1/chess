"""Round-trip and invariant tests for the encoding layer."""

import chess
import numpy as np

from chesszero.encoding import (
    ACTION_SIZE,
    INPUT_PLANES,
    PLANES_TO_FLIP,
    encode_board,
    index_to_move,
    legal_mask,
    move_to_index,
)


def test_action_size():
    assert ACTION_SIZE == 4672


def test_encode_shape_and_side_to_move():
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (INPUT_PLANES, 8, 8)
    assert planes[12].all()  # white to move at the start


def test_repetition_planes():
    board = chess.Board()
    # No repetition yet: both repetition planes are empty.
    planes = encode_board(board)
    assert not planes[18].any()
    assert not planes[19].any()

    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for move in cycle:  # one cycle -> starting position seen twice
        board.push_uci(move)
    planes = encode_board(board)
    assert planes[18].all()       # two-fold: the one-step draw warning
    assert not planes[19].any()   # not yet three-fold

    for move in cycle:  # second cycle -> seen three times
        board.push_uci(move)
    planes = encode_board(board)
    assert planes[18].all()
    assert planes[19].all()       # three-fold


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


def test_black_move_indices_use_inverted_coordinates():
    """Black moves must share the white-to-move canonical orientation."""
    board = chess.Board()
    board.push_uci("e2e4")
    move = chess.Move.from_uci("e7e5")

    # e7 (6, 4) rotates to d2 (1, 3); its two-square advance is plane 1.
    index = move_to_index(move, invert=board.turn == chess.BLACK)
    assert index == 64 + chess.D2
    assert index_to_move(index, board) == move

    mask = legal_mask(board)
    assert mask[index]


def test_board_is_flipped_correctly():
    fens = [
        ("8/2n5/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 10 20", "rnbkqbnr/pppppppp/8/8/8/8/5N2/8 b - - 10 20")
    ]

    for fen in fens:
        boardA = chess.Board(fen[0])
        planesA = encode_board(boardA)
        boardB = chess.Board(fen[1])
        planesB = encode_board(boardB)

        for i in PLANES_TO_FLIP:

            if not np.array_equal(planesA[i], planesB[i]):
                print(f"\nMismatch in plane {i}")

                mask = planesA[i] != planesB[i]

                for row, col in np.argwhere(mask):
                    print(
                        f"  position=({row}, {col}): "
                        f"A={planesA[i, row, col]}, "
                        f"B={planesB[i, row, col]}"
                    )

                assert False

    

def test_legal_mask_matches_legal_moves():
    board = chess.Board()
    mask = legal_mask(board)
    assert mask.sum() == board.legal_moves.count()
    for move in board.legal_moves:
        assert mask[move_to_index(move)]


def test_indices_are_unique_per_position():
    board = chess.Board()
    indices = [move_to_index(m) for m in board.legal_moves]
    assert len(indices) == len(set(indices))
