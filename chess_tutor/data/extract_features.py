"""Extract handcrafted board-state and move features from chess positions."""

import chess
import numpy as np

from ..config import PIECE_VALUES


def compute_material_balance(board: chess.Board) -> float:
    """Returns material balance in centipawns (positive = white advantage)."""
    balance = 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        balance += len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt]
        balance -= len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]
    return float(balance)


def compute_mobility(board: chess.Board) -> int:
    """Returns number of legal moves for side to move."""
    return board.legal_moves.count()


def compute_king_safety(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    """Returns (pawn_shield_score, king_zone_attacks)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0, 0

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)

    shield_count = 0
    for f in [max(0, king_file - 1), king_file, min(7, king_file + 1)]:
        if 0 <= shield_rank <= 7:
            sq = chess.square(f, shield_rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                shield_count += 1

    # King zone attacks
    king_zone = set(board.attacks(king_sq))
    king_zone.add(king_sq)
    attacks = sum(1 for sq in king_zone if board.is_attacked_by(not color, sq))

    return shield_count, min(attacks, 8)


def compute_center_control(board: chess.Board) -> float:
    """Returns center control score."""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    extended = [chess.C3, chess.C4, chess.C5, chess.C6,
                chess.D3, chess.D6, chess.E3, chess.E6,
                chess.F3, chess.F4, chess.F5, chess.F6]
    score = 0.0
    color = board.turn
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            score += 1.0
        if board.is_attacked_by(color, sq):
            score += 0.5
    for sq in extended:
        if board.is_attacked_by(color, sq):
            score += 0.1
    return score


def compute_pawn_structure(board: chess.Board, color: chess.Color) -> tuple[int, int, int, int]:
    """Returns (isolated, doubled, passed, islands) pawn counts."""
    pawns = board.pieces(chess.PAWN, color)
    if not pawns:
        return 0, 0, 0, 0

    files_with_pawns = set()
    for sq in pawns:
        files_with_pawns.add(chess.square_file(sq))

    # Isolated pawns
    isolated = 0
    for sq in pawns:
        f = chess.square_file(sq)
        has_neighbor = False
        for adj_f in [f - 1, f + 1]:
            if adj_f in files_with_pawns:
                has_neighbor = True
                break
        if not has_neighbor:
            isolated += 1

    # Doubled pawns
    file_counts = {}
    for sq in pawns:
        f = chess.square_file(sq)
        file_counts[f] = file_counts.get(f, 0) + 1
    doubled = sum(c - 1 for c in file_counts.values() if c > 1)

    # Passed pawns
    opp_pawns = board.pieces(chess.PAWN, not color)
    opp_files = {}
    for sq in opp_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if f not in opp_files:
            opp_files[f] = []
        opp_files[f].append(r)

    passed = 0
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for check_f in [f - 1, f, f + 1]:
            if check_f in opp_files:
                for opp_r in opp_files[check_f]:
                    if color == chess.WHITE and opp_r > r:
                        is_passed = False
                    elif color == chess.BLACK and opp_r < r:
                        is_passed = False
        if is_passed:
            passed += 1

    # Pawn islands
    sorted_files = sorted(files_with_pawns)
    islands = 1 if sorted_files else 0
    for i in range(1, len(sorted_files)):
        if sorted_files[i] - sorted_files[i - 1] > 1:
            islands += 1

    return isolated, doubled, passed, islands


def compute_development(board: chess.Board, color: chess.Color) -> int:
    """Returns number of minor pieces developed (off starting squares)."""
    if color == chess.WHITE:
        starting = {
            chess.KNIGHT: [chess.B1, chess.G1],
            chess.BISHOP: [chess.C1, chess.F1],
        }
    else:
        starting = {
            chess.KNIGHT: [chess.B8, chess.G8],
            chess.BISHOP: [chess.C8, chess.F8],
        }
    developed = 0
    for pt, start_sqs in starting.items():
        for sq in board.pieces(pt, color):
            if sq not in start_sqs:
                developed += 1
    return min(developed, 4)


def detect_game_phase(board: chess.Board) -> str:
    """Returns 'opening', 'middlegame', or 'endgame' based on material."""
    total = 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        total += len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt]
        total += len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]

    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))

    if total > 6000 and queens >= 2:
        return "opening" if board.fullmove_number <= 10 else "middlegame"
    elif total < 2500 or queens == 0:
        return "endgame"
    else:
        return "middlegame"


def count_hanging_pieces(board: chess.Board, color: chess.Color) -> int:
    """Returns number of undefended pieces under attack."""
    hanging = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                hanging += 1
    return hanging


def extract_board_features(board: chess.Board) -> np.ndarray:
    """Extract handcrafted board-state features from a chess position.

    Returns:
        np.ndarray of shape (30,) -- float64
    """
    features = np.zeros(30, dtype=np.float64)

    # [0:6] White piece counts
    for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
        features[i] = len(board.pieces(pt, chess.WHITE))

    # [6:12] Black piece counts
    for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
        features[6 + i] = len(board.pieces(pt, chess.BLACK))

    color = board.turn

    features[12] = compute_material_balance(board)
    features[13] = compute_mobility(board)

    shield, attacks = compute_king_safety(board, color)
    features[14] = shield
    features[15] = attacks

    features[16] = compute_center_control(board)

    isolated, doubled, passed, islands = compute_pawn_structure(board, color)
    features[17] = isolated
    features[18] = doubled
    features[19] = passed
    features[20] = islands

    features[21] = compute_development(board, color)

    # [22:26] Castling rights
    features[22] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[23] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[24] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[25] = float(board.has_queenside_castling_rights(chess.BLACK))

    # [26:29] Game phase one-hot
    phase = detect_game_phase(board)
    features[26] = float(phase == "opening")
    features[27] = float(phase == "middlegame")
    features[28] = float(phase == "endgame")

    features[29] = count_hanging_pieces(board, color)

    return features


def extract_move_features(
    board: chess.Board,
    move: chess.Move,
    stockfish_eval_before: float | None = None,
    stockfish_eval_after: float | None = None,
) -> np.ndarray:
    """Extract features describing a specific move in context.

    Returns:
        np.ndarray of shape (10,) -- float64
    """
    features = np.zeros(10, dtype=np.float64)

    features[0] = float(board.is_capture(move))

    # Check if the move gives check
    board_copy = board.copy()
    board_copy.push(move)
    features[1] = float(board_copy.is_check())

    # [2:8] Piece type one-hot
    piece = board.piece_at(move.from_square)
    if piece:
        idx = piece.piece_type - 1  # PAWN=1 -> idx 0, etc.
        if 0 <= idx < 6:
            features[2 + idx] = 1.0

    # Centipawn loss
    if stockfish_eval_before is not None and stockfish_eval_after is not None:
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        eval_diff = sign * (stockfish_eval_after - stockfish_eval_before)
        features[8] = max(0.0, -eval_diff)
    else:
        features[8] = 0.0

    # Normalized move number
    features[9] = board.fullmove_number / 80.0

    return features


def extract_all_features(
    board: chess.Board,
    move: chess.Move,
    sf_eval_before: float | None = None,
    sf_eval_after: float | None = None,
) -> np.ndarray:
    """Concatenate board + move features.

    Returns:
        np.ndarray of shape (40,)
    """
    return np.concatenate([
        extract_board_features(board),
        extract_move_features(board, move, sf_eval_before, sf_eval_after),
    ])
