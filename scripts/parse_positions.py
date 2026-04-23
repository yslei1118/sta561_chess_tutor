"""Parse a raw Lichess PGN into ``data/processed/parsed_positions.parquet``."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from chess_tutor.data.parse_pgn import parse_pgn_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/lichess_2014_01.pgn",
        help="Path to the decompressed PGN file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/parsed_positions.parquet",
        help="Destination parquet path.",
    )
    parser.add_argument(
        "--max-games-per-bracket",
        type=int,
        default=100_000,
        help="Maximum number of games to keep in each ELO bracket.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = parse_pgn_file(args.input, max_games_per_bracket=args.max_games_per_bracket)
    df.to_parquet(args.output)
    print(f"Saved: {args.output} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
