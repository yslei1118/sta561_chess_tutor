"""Download Lichess PGN files and pre-computed evaluations."""

import os
import urllib.request

try:
    import zstandard
except ImportError:
    zstandard = None


def download_lichess_pgn(year: int, month: int, output_dir: str = "data/raw/") -> str:
    """Download and decompress a monthly Lichess PGN file.

    Args:
        year: e.g. 2023
        month: e.g. 1 (January)
        output_dir: where to save the .pgn file

    Returns:
        Path to the decompressed .pgn file
    """
    pgn_path = os.path.join(output_dir, f"lichess_{year}_{month:02d}.pgn")
    if os.path.exists(pgn_path):
        return pgn_path

    os.makedirs(output_dir, exist_ok=True)

    url = (
        f"https://database.lichess.org/standard/"
        f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    )
    zst_path = os.path.join(output_dir, f"lichess_{year}_{month:02d}.pgn.zst")

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zst_path)

    if zstandard is None:
        raise ImportError("zstandard is required for decompression: pip install zstandard")

    print("Decompressing ...")
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as ifh, open(pgn_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)

    os.remove(zst_path)
    print(f"Saved to {pgn_path}")
    return pgn_path


def download_stockfish_evals(output_dir: str = "data/raw/") -> str:
    """Download pre-computed Stockfish evaluations from HuggingFace.

    Returns:
        Path to the downloaded parquet file(s)
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "lichess_evals.parquet")
    if os.path.exists(out_path):
        return out_path

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="Lichess/chess-position-evaluations",
            filename="data/train-00000-of-00016.parquet",
            repo_type="dataset",
            local_dir=output_dir,
        )
        return path
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("You can manually download from https://huggingface.co/datasets/Lichess/chess-position-evaluations")
        return out_path


def download_puzzles(output_dir: str = "data/raw/") -> str:
    """Download Lichess puzzle database.

    Returns:
        Path to puzzles CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "lichess_puzzles.csv")
    if os.path.exists(csv_path):
        return csv_path

    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    zst_path = csv_path + ".zst"

    print(f"Downloading puzzles from {url} ...")
    urllib.request.urlretrieve(url, zst_path)

    if zstandard is None:
        raise ImportError("zstandard is required: pip install zstandard")

    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as ifh, open(csv_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)

    os.remove(zst_path)
    print(f"Saved to {csv_path}")
    return csv_path
