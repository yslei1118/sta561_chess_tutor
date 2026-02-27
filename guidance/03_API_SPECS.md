# Chess Tutor: API Specifications

> Every public function and class that Claude Code must implement. Types use Python type hints.

---

## §1 Data Module (`chess_tutor/data/`)

### download.py

```python
def download_lichess_pgn(year: int, month: int, output_dir: str = "data/raw/") -> str:
    """Download and decompress a monthly Lichess PGN file.
    
    Args:
        year: e.g. 2023
        month: e.g. 1 (January)
        output_dir: where to save the .pgn file
    
    Returns:
        Path to the decompressed .pgn file
    
    Notes:
        - Downloads from https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst
        - Uses zstandard library for decompression
        - Streams to disk (don't load entire file into memory)
        - Skip if file already exists
    """

def download_stockfish_evals(output_dir: str = "data/raw/") -> str:
    """Download pre-computed Stockfish evaluations from HuggingFace.
    
    Returns:
        Path to the downloaded parquet file(s)
    
    Notes:
        - Dataset: Lichess/chess-position-evaluations on HuggingFace
        - Use huggingface_hub or direct URL download
        - These are 342M positions — consider downloading a subset
    """

def download_puzzles(output_dir: str = "data/raw/") -> str:
    """Download Lichess puzzle database.
    
    Returns:
        Path to puzzles CSV file
    
    Notes:
        - From https://database.lichess.org/puzzles
        - ~1GB CSV with columns: PuzzleId, FEN, Moves, Rating, Themes, etc.
    """
```

### parse_pgn.py

```python
import chess
import chess.pgn
import pandas as pd

def parse_pgn_file(
    pgn_path: str,
    elo_brackets: list[int] = [1100, 1300, 1500, 1700, 1900],
    elo_tolerance: int = 50,
    max_games_per_bracket: int = 100000,
    positions_per_game: int = 10,
    min_move: int = 5,
    max_move: int = 40,
    time_controls: list[str] | None = None,  # None = all standard/rapid
) -> pd.DataFrame:
    """Parse PGN file into per-position records.
    
    Args:
        pgn_path: path to .pgn file
        elo_brackets: target ELO centers
        elo_tolerance: ±range around each bracket center
        max_games_per_bracket: cap per bracket to balance dataset
        positions_per_game: how many positions to sample per game
        min_move, max_move: move number range to sample from
        time_controls: filter for specific time controls (None = standard+rapid)
    
    Returns:
        DataFrame with columns:
            game_id, move_number, fen, white_elo, black_elo, side_to_move,
            player_elo, move_uci, move_san, result, time_control
    
    Notes:
        - Sample positions uniformly within [min_move, max_move]
        - Skip games with missing ELO or abandoned games
        - Both sides' moves are valid samples (one per position)
        - Track games per bracket; stop when max_games_per_bracket reached for all
    """

def parse_single_game(game: chess.pgn.Game) -> list[dict]:
    """Extract position records from a single game.
    
    Returns:
        List of dicts, each representing one position in the game
    """
```

### extract_features.py

```python
import chess
import numpy as np

def extract_board_features(board: chess.Board) -> np.ndarray:
    """Extract handcrafted board-state features from a chess position.
    
    Args:
        board: python-chess Board object
    
    Returns:
        np.ndarray of shape (30,) — float64
    
    Feature breakdown (30 total):
        [0:6]   White piece counts: P, N, B, R, Q, K
        [6:12]  Black piece counts: P, N, B, R, Q, K
        [12]    Material balance (centipawn equiv, white perspective)
        [13]    Mobility: number of legal moves for side to move
        [14]    King safety: pawn shield score (0-3)
        [15]    King safety: attacks on king zone (0-8)
        [16]    Center control score (pieces/pawns on central squares)
        [17]    Isolated pawns count (side to move)
        [18]    Doubled pawns count (side to move)
        [19]    Passed pawns count (side to move)
        [20]    Pawn islands count (side to move)
        [21]    Development score: minor pieces off starting squares (0-4)
        [22:26] Castling rights: white_king, white_queen, black_king, black_queen (binary)
        [26:29] Game phase one-hot: opening, middlegame, endgame
        [29]    Hanging pieces count (side to move)
    """

def extract_move_features(board: chess.Board, move: chess.Move, 
                          stockfish_eval_before: float | None = None,
                          stockfish_eval_after: float | None = None) -> np.ndarray:
    """Extract features describing a specific move in context.
    
    Args:
        board: position before the move
        move: the move being analyzed
        stockfish_eval_before: eval before move (optional, in centipawns)
        stockfish_eval_after: eval after move (optional)
    
    Returns:
        np.ndarray of shape (10,) — float64
    
    Feature breakdown (10 total):
        [0]     Is capture (binary)
        [1]     Is check (binary)
        [2:8]   Piece type one-hot: P, N, B, R, Q, K
        [8]     Centipawn loss from best move (0 if unavailable)
        [9]     Normalized move number (move_number / 80)
    """

def extract_all_features(board: chess.Board, move: chess.Move,
                         sf_eval_before: float | None = None,
                         sf_eval_after: float | None = None) -> np.ndarray:
    """Concatenate board + move features.
    
    Returns:
        np.ndarray of shape (40,)
    """
    # return np.concatenate([extract_board_features(board), 
    #                        extract_move_features(board, move, sf_eval_before, sf_eval_after)])

# Helper functions (internal, but test individually)
def compute_material_balance(board: chess.Board) -> float:
    """Returns material balance in centipawns (positive = white advantage)."""

def compute_mobility(board: chess.Board) -> int:
    """Returns number of legal moves for side to move."""

def compute_king_safety(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    """Returns (pawn_shield_score, king_zone_attacks)."""

def compute_center_control(board: chess.Board) -> float:
    """Returns center control score."""

def compute_pawn_structure(board: chess.Board, color: chess.Color) -> tuple[int, int, int, int]:
    """Returns (isolated, doubled, passed, islands) pawn counts."""

def compute_development(board: chess.Board, color: chess.Color) -> int:
    """Returns number of minor pieces developed (off starting squares)."""

def detect_game_phase(board: chess.Board) -> str:
    """Returns 'opening', 'middlegame', or 'endgame' based on material."""

def count_hanging_pieces(board: chess.Board, color: chess.Color) -> int:
    """Returns number of undefended pieces under attack."""
```

### stockfish_eval.py

```python
import chess
import chess.engine

class StockfishEvaluator:
    """Wrapper for Stockfish position evaluation."""
    
    def __init__(self, stockfish_path: str = "stockfish", depth: int = 15):
        """Initialize Stockfish engine.
        
        Args:
            stockfish_path: path to stockfish binary
            depth: search depth for evaluation
        """
    
    def evaluate(self, board: chess.Board) -> dict:
        """Evaluate a position.
        
        Returns:
            {
                'score_cp': float,       # Centipawn score (white perspective)
                'best_move': str,        # Best move in UCI format
                'best_line': list[str],  # Principal variation (top 3 moves)
                'is_mate': bool,         # Whether position is mate
                'mate_in': int | None,   # Moves to mate (if applicable)
            }
        """
    
    def evaluate_move(self, board: chess.Board, move: chess.Move) -> dict:
        """Evaluate a specific move.
        
        Returns:
            {
                'cp_loss': float,        # Centipawn loss from best move (≥0)
                'is_blunder': bool,      # cp_loss > BLUNDER_THRESHOLD
                'is_mistake': bool,      # cp_loss > 50
                'is_inaccuracy': bool,   # cp_loss > 20
                'eval_before': float,    # Eval before move
                'eval_after': float,     # Eval after move
            }
        """
    
    def batch_evaluate(self, positions: list[chess.Board], 
                       batch_size: int = 100) -> list[dict]:
        """Evaluate multiple positions efficiently."""
    
    def close(self):
        """Close the engine process."""
    
    def __enter__(self): ...
    def __exit__(self, *args): ...
```

### dataset.py

```python
import pandas as pd
import numpy as np
from pathlib import Path

class ChessTutorDataset:
    """Main dataset class for the chess tutor project."""
    
    def __init__(self, data_dir: str = "data/processed/"):
        """Initialize with path to processed data directory."""
        self.data_dir = Path(data_dir)
        self.positions: pd.DataFrame | None = None
        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
    
    def build_from_pgn(self, pgn_paths: list[str], 
                       stockfish_path: str = "stockfish",
                       use_hf_evals: bool = True) -> None:
        """Full pipeline: PGN → parsed positions → features → save.
        
        Args:
            pgn_paths: list of PGN file paths
            stockfish_path: path to stockfish binary
            use_hf_evals: if True, use HuggingFace pre-computed evals instead of running Stockfish
        """
    
    def save(self, filename: str = "chess_tutor_dataset") -> None:
        """Save dataset to parquet files in data_dir."""
    
    def load(self, filename: str = "chess_tutor_dataset") -> None:
        """Load dataset from parquet files."""
    
    def get_bracket(self, elo: int, tolerance: int = 50) -> pd.DataFrame:
        """Get positions for a specific ELO bracket."""
    
    def train_test_split(self, test_month: int | None = None, 
                         test_frac: float = 0.2,
                         random_state: int = 42) -> tuple:
        """Split into train/test.
        
        Args:
            test_month: if provided, use temporal split (this month = test)
            test_frac: if test_month is None, use random split with this fraction
        
        Returns:
            (X_train, y_train, meta_train, X_test, y_test, meta_test)
            where X = feature arrays, y = move labels, meta = DataFrame with game_id, elo, etc.
        """
    
    def get_features_and_labels(self, elo_bracket: int | None = None
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Get feature matrix X and label vector y.
        
        Args:
            elo_bracket: if provided, filter to this bracket
        
        Returns:
            (X, y) where X is (n_samples, n_features) and y is (n_samples,) move indices
        """
    
    @property
    def elo_brackets(self) -> list[int]:
        """Return sorted list of ELO brackets present in dataset."""
    
    @property
    def n_positions(self) -> int:
        """Total number of positions."""
    
    def summary(self) -> dict:
        """Return summary statistics of the dataset."""
```

---

## §2 Models Module (`chess_tutor/models/`)

### move_predictor.py

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class MovePredictor:
    """ELO-conditioned human move predictor.
    
    Supports three architectures:
        A: Separate model per ELO bracket (Maia-1 style)
        B: Single pooled model with ELO as feature
        C: Kernel-smoothed interpolation across brackets
    """
    
    def __init__(self, architecture: str = 'C',
                 base_classifier: str = 'rf',
                 elo_brackets: list[int] = [1100, 1300, 1500, 1700, 1900],
                 kernel_bandwidth: float = 100.0,
                 **classifier_kwargs):
        """
        Args:
            architecture: 'A', 'B', or 'C'
            base_classifier: 'rf', 'svm', 'logistic', 'ridge'
            elo_brackets: ELO bracket centers (for arch A and C)
            kernel_bandwidth: Gaussian kernel bandwidth for arch C (in ELO points)
            **classifier_kwargs: passed to sklearn classifier constructor
        """
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            elos: np.ndarray, **fit_kwargs) -> 'MovePredictor':
        """Train the model.
        
        Args:
            X: feature matrix (n_samples, n_features)
            y: move labels (n_samples,) — integer indices into move vocabulary
            elos: player ELO for each sample (n_samples,)
        
        Returns:
            self (for chaining)
        
        Notes:
            - Architecture A: trains one classifier per bracket
            - Architecture B: appends normalized ELO to X, trains single classifier
            - Architecture C: trains per-bracket models (like A), stores for kernel combination
        """
    
    def predict(self, X: np.ndarray, target_elo: int, 
                top_k: int = 5) -> np.ndarray:
        """Predict move probabilities for a target ELO.
        
        Args:
            X: feature matrix (n_samples, n_features)
            target_elo: the ELO level to predict for
            top_k: return top-K predictions
        
        Returns:
            Array of shape (n_samples, top_k) with move indices sorted by probability
        """
    
    def predict_proba(self, X: np.ndarray, target_elo: int) -> np.ndarray:
        """Get full probability distribution over moves.
        
        Args:
            X: feature matrix (n_samples, n_features)
            target_elo: ELO level
        
        Returns:
            (n_samples, n_moves) probability matrix
        
        Notes:
            - Architecture C uses Nadaraya-Watson kernel combination:
              P(m|x, elo*) = Σ_k K(elo* - elo_k) · P_k(m|x) / Σ_k K(elo* - elo_k)
        """
    
    def score(self, X: np.ndarray, y: np.ndarray, elos: np.ndarray) -> dict:
        """Evaluate model performance.
        
        Returns:
            {
                'accuracy_top1': float,
                'accuracy_top3': float,
                'accuracy_top5': float,
                'per_bracket_accuracy': dict[int, float],
                'cross_elo_matrix': np.ndarray,  # (n_brackets, n_brackets) accuracy matrix
            }
        """
    
    def cross_elo_evaluation(self, X_test: np.ndarray, y_test: np.ndarray,
                             test_elos: np.ndarray) -> np.ndarray:
        """Build cross-ELO evaluation matrix.
        
        Returns:
            Matrix of shape (n_brackets, n_brackets) where entry [i,j] is
            accuracy when model trained on bracket i predicts bracket j's moves.
            Diagonal should be highest (model matches test ELO).
        """
    
    def feature_importance(self, bracket: int | None = None) -> dict[str, float]:
        """Get feature importance scores.
        
        Works for RF (Gini importance) and linear models (coefficient magnitude).
        """
    
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### kernel_interpolation.py

```python
import numpy as np

class NadarayaWatsonELO:
    """Nadaraya-Watson kernel regression for interpolating move predictions across ELO.
    
    Given per-bracket classifiers P_k(m|x) at ELO centers s_k, computes:
        P(m|x, s*) = Σ_k K_h(s* - s_k) · P_k(m|x) / Σ_k K_h(s* - s_k)
    
    where K_h is a Gaussian kernel with bandwidth h.
    """
    
    def __init__(self, bandwidth: float = 100.0):
        """
        Args:
            bandwidth: Gaussian kernel bandwidth in ELO points.
                       Small h → sharp interpolation (near-bracket only)
                       Large h → smooth interpolation (averages many brackets)
        """
    
    def kernel(self, elo_query: float, elo_center: float) -> float:
        """Gaussian kernel: K_h(s* - s_k) = exp(-0.5 * ((s* - s_k)/h)^2)"""
    
    def interpolate(self, bracket_predictions: dict[int, np.ndarray],
                    target_elo: float) -> np.ndarray:
        """Combine per-bracket predictions via kernel weighting.
        
        Args:
            bracket_predictions: {elo_center: P_k(m|x)} probability arrays
            target_elo: query ELO
        
        Returns:
            Interpolated probability distribution over moves
        """
    
    def select_bandwidth_cv(self, X: np.ndarray, y: np.ndarray,
                            elos: np.ndarray,
                            bracket_models: dict,
                            bandwidths: list[float] = [25, 50, 100, 150, 200, 300]
                            ) -> float:
        """Select optimal bandwidth via leave-one-bracket-out CV.
        
        For each bracket b:
            1. Hold out bracket b's test data
            2. Interpolate from remaining brackets
            3. Measure accuracy on held-out bracket
        
        Returns:
            Optimal bandwidth (argmax avg accuracy across held-out brackets)
        """
    
    def kernel_weights(self, target_elo: float, 
                       bracket_centers: list[int]) -> np.ndarray:
        """Return normalized kernel weights for each bracket.
        
        Useful for visualization: shows how much each bracket contributes.
        """
```

### position_eval.py

```python
import numpy as np
import chess

class BlunderDetector:
    """Binary classifier: predicts whether a move is a blunder (>100cp loss)."""
    
    def __init__(self, classifier: str = 'rf', **kwargs):
        """
        Args:
            classifier: 'rf', 'gbt', 'logistic'
        """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlunderDetector':
        """Train blunder detector.
        
        Args:
            X: features (board + move features, shape (n, 40))
            y: binary labels (1 = blunder, 0 = not)
        """
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(blunder) for each sample. Shape (n,)."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Returns {'auc': float, 'precision': float, 'recall': float, 'f1': float}"""


class PositionComplexity:
    """Regression model: predicts how difficult a position is for a given ELO."""
    
    def __init__(self, model: str = 'rf'):
        """
        Args:
            model: 'rf', 'ridge', 'gradient_boosting'
        """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PositionComplexity':
        """Train complexity estimator.
        
        Args:
            X: board features (n, 30)
            y: difficulty scores (stddev of cp_loss at that ELO bracket)
        """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return complexity scores. Shape (n,)."""
```

---

## §3 Feedback Module (`chess_tutor/feedback/`)

### taxonomy.py

```python
from enum import IntEnum

class FeedbackType(IntEnum):
    TACTICAL_ALERT = 0       # F1: Tactic available
    STRATEGIC_NUDGE = 1      # F2: Positional improvement
    BLUNDER_WARNING = 2      # F3: High blunder probability
    PATTERN_RECOGNITION = 3  # F4: Known pattern match
    MOVE_COMPARISON = 4      # F5: Compare with engine
    ENCOURAGEMENT = 5        # F6: Good move praise
    SIMPLIFICATION = 6       # F7: Suggest simplifying

# Mapping from feedback type to which chess concepts it targets
FEEDBACK_CONCEPT_MAP: dict[FeedbackType, list[str]] = {
    FeedbackType.TACTICAL_ALERT: ['tactics'],
    FeedbackType.STRATEGIC_NUDGE: ['strategy', 'positional'],
    FeedbackType.BLUNDER_WARNING: ['tactics', 'calculation'],
    FeedbackType.PATTERN_RECOGNITION: ['opening', 'endgame', 'pattern'],
    FeedbackType.MOVE_COMPARISON: ['calculation', 'strategy'],
    FeedbackType.ENCOURAGEMENT: [],  # general reinforcement
    FeedbackType.SIMPLIFICATION: ['endgame', 'strategy'],
}
```

### generator.py

```python
import chess
from .taxonomy import FeedbackType

class FeedbackGenerator:
    """Generate natural language feedback for a position + student ELO."""
    
    def __init__(self, blunder_detector=None, complexity_estimator=None,
                 stockfish_evaluator=None):
        """
        Args:
            blunder_detector: trained BlunderDetector instance
            complexity_estimator: trained PositionComplexity instance
            stockfish_evaluator: StockfishEvaluator instance
        """
    
    def generate(self, board: chess.Board, student_elo: int,
                 feedback_type: FeedbackType,
                 student_move: chess.Move | None = None,
                 engine_eval: dict | None = None) -> str:
        """Generate feedback text.
        
        Args:
            board: current position
            student_elo: student's ELO
            feedback_type: which type of feedback to generate
            student_move: move the student played (if any)
            engine_eval: pre-computed Stockfish evaluation
        
        Returns:
            Natural language feedback string
        """
    
    def select_best_feedback_type(self, board: chess.Board, 
                                   student_elo: int,
                                   engine_eval: dict) -> FeedbackType:
        """Heuristic feedback selection (baseline for bandit comparison).
        
        Rules:
            1. If tactic available and student likely to miss it → TACTICAL_ALERT
            2. If blunder_prob > 0.5 → BLUNDER_WARNING
            3. If position is very complex → SIMPLIFICATION
            4. If student move was good → ENCOURAGEMENT
            5. Default → STRATEGIC_NUDGE
        """
    
    def analyze_position_for_feedback(self, board: chess.Board,
                                       engine_eval: dict) -> dict:
        """Analyze position to determine what feedback is relevant.
        
        Returns:
            {
                'has_tactic': bool,
                'tactic_type': str | None,  # 'fork', 'pin', 'skewer', etc.
                'blunder_prob': float,
                'complexity': float,
                'critical_squares': list[str],
                'key_pieces': list[str],
                'position_type': str,  # 'tactical', 'positional', 'endgame'
            }
        """
```

---

## §4 Teaching Module (`chess_tutor/teaching/`)

### bandit.py

```python
import numpy as np

class LinearThompsonSampling:
    """Thompson Sampling for Linear Contextual Bandits.
    
    Following Agrawal & Goyal (ICML 2013).
    For each arm a, maintains Bayesian linear regression:
        r = θ_a^T x + ε
    with posterior N(μ̂_a, v² B_a^{-1}).
    """
    
    def __init__(self, n_arms: int = 7, context_dim: int = 20,
                 exploration_v: float = 1.0, prior_variance: float = 1.0):
        """
        Args:
            n_arms: number of arms (feedback types)
            context_dim: dimension of context vector
            exploration_v: exploration parameter (controls posterior sampling spread)
            prior_variance: prior variance for θ (isotropic Gaussian)
        """
        # For each arm a:
        #   B_a = I_d (d×d precision matrix, initialized to identity)
        #   f_a = 0_d (d-dim reward accumulator)
        #   μ̂_a = B_a^{-1} f_a (posterior mean)
    
    def select_arm(self, context: np.ndarray) -> int:
        """Select an arm (feedback type) given context.
        
        Args:
            context: (context_dim,) feature vector
        
        Returns:
            Selected arm index (0 to n_arms-1)
        
        Algorithm:
            For each arm a:
                Sample θ̃_a ~ N(μ̂_a, v² B_a^{-1})
                score_a = θ̃_a^T context
            Return argmax_a score_a
        """
    
    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update posterior for the selected arm.
        
        Args:
            arm: which arm was selected
            context: context vector at selection time
            reward: observed reward
        
        Update rules:
            B_arm += context @ context.T
            f_arm += reward * context
            μ̂_arm = B_arm^{-1} @ f_arm
        """
    
    def get_posterior(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, covariance) of posterior for an arm.
        
        Returns:
            (μ̂_a, v² B_a^{-1}) — mean vector and covariance matrix
        """
    
    def get_arm_scores(self, context: np.ndarray) -> np.ndarray:
        """Return expected reward for each arm (posterior mean, no sampling).
        
        Returns:
            (n_arms,) array of μ̂_a^T context
        """
    
    def reset(self) -> None:
        """Reset all posteriors to prior."""
    
    @property
    def total_pulls(self) -> np.ndarray:
        """(n_arms,) count of times each arm was pulled."""
    
    @property
    def cumulative_reward(self) -> float:
        """Total reward accumulated."""


class EpsilonGreedy:
    """ε-Greedy baseline policy."""
    
    def __init__(self, n_arms: int = 7, epsilon: float = 0.1): ...
    def select_arm(self, context: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...


class LinUCB:
    """LinUCB baseline (Li et al., WWW 2010)."""
    
    def __init__(self, n_arms: int = 7, context_dim: int = 20, alpha: float = 1.0): ...
    def select_arm(self, context: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...


class RandomPolicy:
    """Uniform random baseline."""
    
    def __init__(self, n_arms: int = 7): ...
    def select_arm(self, context: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...


class RuleBasedPolicy:
    """Hand-designed heuristic baseline."""
    
    def __init__(self): ...
    def select_arm(self, context: np.ndarray) -> int:
        """Rules:
        - If blunder_prob (context[5]) > 0.5 → BLUNDER_WARNING (2)
        - If has_tactic proxy (context[4] > threshold) → TACTICAL_ALERT (0)
        - If complexity (context[4]) > high_threshold → SIMPLIFICATION (6)
        - Default → STRATEGIC_NUDGE (1)
        """
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...
```

### context_builder.py

```python
import numpy as np
import chess

def build_context(board: chess.Board, student_state: 'StudentState',
                  blunder_prob: float, complexity: float,
                  board_features: np.ndarray | None = None) -> np.ndarray:
    """Build context vector for the contextual bandit.
    
    Args:
        board: current chess position
        student_state: StudentState object with ELO, weakness profile, etc.
        blunder_prob: P(blunder) from BlunderDetector
        complexity: position complexity score
        board_features: pre-computed board features (optional, computed if None)
    
    Returns:
        np.ndarray of shape (20,) — the context vector
    
    Context breakdown (20 dims):
        [0:4]   Board features subset: material_balance, mobility, king_safety_shield, king_safety_attacks
        [4]     Position complexity
        [5]     Blunder probability
        [6]     Student ELO (normalized: (elo - 1100) / 800)
        [7]     Student recent avg centipawn loss (normalized)
        [8:11]  Student weakness profile: tactical, strategic, endgame
        [11]    Normalized move number
        [12]    Time pressure indicator
        [13]    Running blunder count (normalized)
        [14:17] Student trend one-hot: improving, stable, declining
        [17]    Game result probability from position (from stockfish eval → win prob)
        [18:20] Reserved / padding (zero)
    """
```

### reward.py

```python
def compute_reward(cp_loss_before_feedback: float, 
                   cp_loss_after_feedback: float,
                   blunder_avoided: bool,
                   continued_play: bool,
                   weights: dict | None = None) -> float:
    """Compute reward for a teaching interaction.
    
    Args:
        cp_loss_before_feedback: centipawn loss of move before feedback was given
        cp_loss_after_feedback: centipawn loss of move after feedback
        blunder_avoided: whether student avoided a predicted blunder
        continued_play: whether student continued playing (didn't quit)
        weights: {'cp_improvement': α, 'blunder_avoided': β, 'continued_play': γ}
    
    Returns:
        Scalar reward in [0, 1] range (clipped)
    
    Formula:
        r = α · sigmoid(cp_loss_before - cp_loss_after) + β · blunder_avoided + γ · continued_play
        normalized to [0, 1]
    """
```

---

## §5 Simulation Module (`chess_tutor/simulation/`)

### student_simulator.py

```python
import chess
import numpy as np

class StudentSimulator:
    """Simulates a chess student for offline evaluation of teaching policies."""
    
    def __init__(self, elo: int, weakness_profile: dict[str, float] | None = None,
                 move_predictor=None, learning_rate: float = 0.01):
        """
        Args:
            elo: initial ELO
            weakness_profile: {concept: skill_level} where skill_level ∈ [0,1]
                Concepts: 'tactics', 'strategy', 'endgame', 'opening', 'calculation'
                Default: random profile centered around elo-appropriate levels
            move_predictor: trained MovePredictor for sampling moves
            learning_rate: how quickly student improves per useful feedback
        """
    
    def respond_to_position(self, board: chess.Board,
                            feedback_type: int | None = None,
                            feedback_text: str | None = None) -> chess.Move:
        """Student plays a move, possibly influenced by feedback.
        
        Args:
            board: current position
            feedback_type: if feedback was given, which type (0-6)
            feedback_text: the feedback text (for determining relevance)
        
        Returns:
            The move the student plays
        
        Logic:
            1. Sample base move from move_predictor at student's ELO
            2. If feedback was given and is relevant to position:
               - With probability p_learn, upgrade to a better move
               - p_learn = base_rate * relevance * (1 - mastery)
            3. Return the (possibly upgraded) move
        """
    
    def update_state(self, feedback_type: int, move_quality: float,
                     position_concepts: list[str]) -> None:
        """Update student's internal state after a teaching interaction.
        
        Args:
            feedback_type: which feedback was given
            move_quality: centipawn loss of the move played
            position_concepts: which chess concepts this position involves
        
        Updates:
            - weakness_profile: improve relevant concepts if feedback was useful
            - elo: adjust based on running performance
            - recent_cp_losses: append move_quality
        """
    
    @property
    def state(self) -> 'StudentState':
        """Return current StudentState snapshot."""


class StudentPopulation:
    """Generate a population of diverse simulated students."""
    
    @staticmethod
    def generate(n_students: int = 50, 
                 elo_range: tuple[int, int] = (1000, 2000),
                 random_state: int = 42) -> list[StudentSimulator]:
        """Generate diverse students with varying ELOs and weakness profiles."""
```

### episode_runner.py

```python
import chess
from typing import Protocol

class Policy(Protocol):
    """Interface that all bandit policies must implement."""
    def select_arm(self, context: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...

def run_episode(student: StudentSimulator,
                policy: Policy,
                feedback_generator: 'FeedbackGenerator',
                context_builder_fn: callable,
                reward_fn: callable,
                positions: list[chess.Board],
                n_interactions: int = 20) -> dict:
    """Run a single teaching episode.
    
    Args:
        student: simulated student
        policy: bandit policy (Thompson Sampling, UCB, etc.)
        feedback_generator: generates feedback text
        context_builder_fn: builds context vectors
        reward_fn: computes rewards
        positions: list of positions to use for teaching
        n_interactions: how many teaching interactions per episode
    
    Returns:
        {
            'rewards': list[float],
            'cumulative_reward': float,
            'arms_selected': list[int],
            'student_elo_trajectory': list[int],
            'cp_loss_trajectory': list[float],
        }
    """

def run_experiment(students: list[StudentSimulator],
                   policies: dict[str, Policy],
                   n_episodes: int = 100,
                   n_interactions_per_episode: int = 20,
                   **kwargs) -> dict:
    """Run full experiment comparing multiple policies.
    
    Returns:
        {
            policy_name: {
                'mean_cumulative_reward': float,
                'std_cumulative_reward': float,
                'mean_elo_gain': float,
                'regret_curves': np.ndarray,
                'arm_distribution': np.ndarray,
            }
            for policy_name in policies
        }
    """
```

---

## §6 Bot Module (`chess_tutor/bot/`)

### engine.py

```python
import chess
import numpy as np

class ChessTutorBot:
    """Chess bot that plays at a specified ELO level with optional KL regularization."""
    
    def __init__(self, move_predictor=None, stockfish_evaluator=None,
                 kl_lambda: float = 0.0):
        """
        Args:
            move_predictor: trained MovePredictor (architecture C preferred)
            stockfish_evaluator: StockfishEvaluator for KL regularization
            kl_lambda: KL regularization strength (0 = pure human model, ∞ = pure engine)
        """
    
    def play_move(self, board: chess.Board, target_elo: int,
                  temperature: float = 1.0) -> chess.Move:
        """Select a move at the target ELO level.
        
        Args:
            board: current position
            target_elo: desired skill level
            temperature: sampling temperature (lower = more greedy)
        
        Returns:
            Selected move
        
        Algorithm:
            1. Get P_human(m|x, elo) from move_predictor
            2. If kl_lambda > 0: blend with Stockfish
               π_bot(m) ∝ P_human(m) · exp(λ · Q_engine(m))
            3. Sample from π_bot with temperature
        """
    
    def evaluate_position(self, board: chess.Board, target_elo: int) -> dict:
        """Evaluate position from the perspective of a player at target_elo.
        
        Returns:
            {
                'assessment': str,      # 'winning', 'equal', 'losing', 'unclear'
                'confidence': float,     # How confident the assessment is
                'key_features': list,    # Most important features of the position
                'suggested_plan': str,   # Natural language suggestion
                'move_predictions': list, # Top 3 predicted human moves at this ELO
            }
        """
```

### commentary.py

```python
import chess

class CommentaryGenerator:
    """Generate move-by-move commentary for interactive play."""
    
    def __init__(self, feedback_generator=None, move_predictor=None,
                 position_evaluator=None):
        """Initialize with model dependencies."""
    
    def comment_on_student_move(self, board_before: chess.Board,
                                 student_move: chess.Move,
                                 student_elo: int,
                                 engine_eval: dict) -> str:
        """Generate commentary on the student's move.
        
        Returns:
            Commentary string like: "You played Nf3 — a solid developing move!
            The engine slightly prefers d4 here, but Nf3 is a natural choice
            at your level and controls the center well."
        """
    
    def comment_on_bot_move(self, board_before: chess.Board,
                            bot_move: chess.Move,
                            target_elo: int,
                            engine_eval: dict) -> str:
        """Explain why the bot played this move (from bot's 'perspective').
        
        Returns:
            Commentary like: "I played e5 because I want to fight for the center.
            At my level, I often play this move in these positions."
        """
    
    def game_summary(self, moves: list[chess.Move], 
                     evaluations: list[dict],
                     student_elo: int) -> str:
        """Generate end-of-game summary with teaching points."""
```

---

## §7 Evaluation Module (`chess_tutor/evaluation/`)

### metrics.py

```python
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def move_matching_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                           top_k: int = 1) -> float:
    """Fraction of positions where true move is in top-K predictions."""

def centipawn_loss_correlation(predicted_losses: np.ndarray,
                               actual_losses: np.ndarray) -> float:
    """Pearson correlation between predicted and actual centipawn losses."""

def cumulative_regret(rewards: np.ndarray, oracle_reward: float = 1.0) -> np.ndarray:
    """Cumulative regret: Σ(oracle_reward - reward_t)."""

def arm_selection_entropy(arm_counts: np.ndarray) -> float:
    """Shannon entropy of arm selection distribution."""

def elo_gain_rate(elo_trajectory: list[int]) -> float:
    """Rate of ELO improvement (slope of linear fit)."""

def blunder_detection_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AUC for blunder detection."""
```

### ablation.py

```python
def run_ablation_suite(dataset, model_configs: list[dict]) -> pd.DataFrame:
    """Run all ablation experiments.
    
    Ablations:
        1. Model A vs B vs C
        2. Classifier: RF vs SVM vs LogReg vs Ridge
        3. Feature subsets: material → +mobility → +king_safety → full
        4. Data size: 10K → 50K → 100K → 500K
        5. Kernel bandwidth: [25, 50, 100, 150, 200, 300]
        6. Bandit: Thompson vs ε-Greedy vs UCB vs Random vs RuleBased
    
    Returns:
        DataFrame with columns: experiment, config, metric, value
    """
```
