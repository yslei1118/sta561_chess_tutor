"""Parse PGN files into per-position records."""

import chess
import chess.pgn
import pandas as pd

from ..config import (
    ELO_BRACKETS, ELO_BRACKET_WIDTH, MIN_MOVE_NUMBER, MAX_MOVE_NUMBER,
    MOVE_SAMPLE_INTERVAL, POSITIONS_PER_GAME,
)


def parse_single_game(game: chess.pgn.Game) -> list[dict]:
    """Extract position records from a single game.

    Returns:
        List of dicts, each representing one position in the game.
    """
    headers = game.headers
    white_elo = int(headers.get("WhiteElo", 0))
    black_elo = int(headers.get("BlackElo", 0))
    result = headers.get("Result", "*")
    time_control = headers.get("TimeControl", "")
    game_id = headers.get("Site", headers.get("Event", ""))

    records = []
    board = game.board()
    move_num = 0

    for node in game.mainline():
        move_num += 1
        move = node.move

        if MIN_MOVE_NUMBER <= move_num <= MAX_MOVE_NUMBER:
            if move_num % MOVE_SAMPLE_INTERVAL == 0:
                player_elo = white_elo if board.turn == chess.WHITE else black_elo
                records.append({
                    "game_id": game_id,
                    "move_number": move_num,
                    "fen": board.fen(),
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "side_to_move": "white" if board.turn == chess.WHITE else "black",
                    "player_elo": player_elo,
                    "move_uci": move.uci(),
                    "move_san": board.san(move),
                    "result": result,
                    "time_control": time_control,
                })

        board.push(move)

    return records


def parse_pgn_file(
    pgn_path: str,
    elo_brackets: list[int] | None = None,
    elo_tolerance: int = ELO_BRACKET_WIDTH,
    max_games_per_bracket: int = 100000,
    positions_per_game: int = POSITIONS_PER_GAME,
    min_move: int = MIN_MOVE_NUMBER,
    max_move: int = MAX_MOVE_NUMBER,
    time_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Parse PGN file into per-position records.

    Returns:
        DataFrame with columns: game_id, move_number, fen, white_elo, black_elo,
        side_to_move, player_elo, move_uci, move_san, result, time_control
    """
    if elo_brackets is None:
        elo_brackets = list(ELO_BRACKETS)

    records = []
    bracket_counts = {b: 0 for b in elo_brackets}

    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            headers = game.headers
            try:
                white_elo = int(headers.get("WhiteElo", "0"))
                black_elo = int(headers.get("BlackElo", "0"))
            except (ValueError, TypeError):
                continue

            if white_elo == 0 or black_elo == 0:
                continue

            # Filter terminated/abandoned
            if headers.get("Termination", "") == "Abandoned":
                continue

            # Time control filter
            tc = headers.get("TimeControl", "")
            if time_controls is not None and tc not in time_controls:
                continue

            # Check bracket
            avg_elo = (white_elo + black_elo) / 2
            target_bracket = None
            for b in elo_brackets:
                if abs(avg_elo - b) <= elo_tolerance:
                    if bracket_counts[b] < max_games_per_bracket:
                        target_bracket = b
                    break

            if target_bracket is None:
                continue

            bracket_counts[target_bracket] += 1

            # Walk through game
            board = game.board()
            move_num = 0
            sampled = 0

            for node in game.mainline():
                move_num += 1
                move = node.move

                if min_move <= move_num <= max_move and sampled < positions_per_game:
                    if move_num % MOVE_SAMPLE_INTERVAL == 0 or (
                        positions_per_game > (max_move - min_move) // MOVE_SAMPLE_INTERVAL
                    ):
                        player_elo = white_elo if board.turn == chess.WHITE else black_elo
                        records.append({
                            "game_id": headers.get("Site", ""),
                            "move_number": move_num,
                            "fen": board.fen(),
                            "white_elo": white_elo,
                            "black_elo": black_elo,
                            "side_to_move": "white" if board.turn == chess.WHITE else "black",
                            "player_elo": player_elo,
                            "move_uci": move.uci(),
                            "move_san": board.san(move),
                            "result": headers.get("Result", "*"),
                            "time_control": tc,
                        })
                        sampled += 1

                board.push(move)

            # Check if all brackets full
            if all(c >= max_games_per_bracket for c in bracket_counts.values()):
                break

    return pd.DataFrame(records)
