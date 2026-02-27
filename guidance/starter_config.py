# chess_tutor/config.py
"""Global constants and configuration for the Chess Tutor project."""

# =============================================================================
# ELO Configuration
# =============================================================================
ELO_BRACKETS = [1100, 1300, 1500, 1700, 1900]
ELO_BRACKET_WIDTH = 50  # ±50 around center
ELO_MIN = 800
ELO_MAX = 2200

# =============================================================================
# Feature Dimensions
# =============================================================================
N_BOARD_FEATURES = 30
N_MOVE_FEATURES = 10
N_TOTAL_FEATURES = N_BOARD_FEATURES + N_MOVE_FEATURES  # 40
N_CONTEXT_FEATURES = 20  # for contextual bandit

# Feature names (for interpretability and plotting)
BOARD_FEATURE_NAMES = [
    'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens', 'white_kings',
    'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens', 'black_kings',
    'material_balance_cp', 'mobility', 'king_safety_shield', 'king_safety_attacks',
    'center_control', 'isolated_pawns', 'doubled_pawns', 'passed_pawns', 'pawn_islands',
    'development_score', 'castling_wk', 'castling_wq', 'castling_bk', 'castling_bq',
    'phase_opening', 'phase_middlegame', 'phase_endgame', 'hanging_pieces'
]

MOVE_FEATURE_NAMES = [
    'is_capture', 'is_check', 'piece_pawn', 'piece_knight', 'piece_bishop',
    'piece_rook', 'piece_queen', 'piece_king', 'cp_loss', 'normalized_move_num'
]

CONTEXT_FEATURE_NAMES = [
    'material_balance', 'mobility', 'king_safety_shield', 'king_safety_attacks',
    'position_complexity', 'blunder_probability', 'student_elo_normalized',
    'student_recent_cp_loss', 'weakness_tactical', 'weakness_strategic', 'weakness_endgame',
    'normalized_move_num', 'time_pressure', 'running_blunder_count',
    'trend_improving', 'trend_stable', 'trend_declining',
    'game_result_prob', 'reserved_1', 'reserved_2'
]

# =============================================================================
# Piece Values (centipawns)
# =============================================================================
PIECE_VALUES = {
    1: 100,   # PAWN
    2: 320,   # KNIGHT
    3: 330,   # BISHOP
    4: 500,   # ROOK
    5: 900,   # QUEEN
    6: 0,     # KING
}

# =============================================================================
# Feedback Configuration
# =============================================================================
N_FEEDBACK_TYPES = 7
FEEDBACK_TYPE_NAMES = [
    'Tactical Alert',       # F1
    'Strategic Nudge',      # F2
    'Blunder Warning',      # F3
    'Pattern Recognition',  # F4
    'Move Comparison',      # F5
    'Encouragement',        # F6
    'Simplification',       # F7
]

# =============================================================================
# Model Hyperparameters
# =============================================================================
# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 5
RF_N_JOBS = -1
RF_RANDOM_STATE = 42

# SVM
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'

# Kernel Interpolation
KERNEL_BANDWIDTH_DEFAULT = 100.0  # ELO points
KERNEL_BANDWIDTH_CANDIDATES = [25, 50, 75, 100, 150, 200, 300]

# =============================================================================
# Bandit Hyperparameters
# =============================================================================
TS_EXPLORATION_V = 1.0
TS_PRIOR_VARIANCE = 1.0
LINUCB_ALPHA = 1.0
EPSILON_GREEDY_EPS = 0.1

# Reward weights
REWARD_WEIGHTS = {
    'cp_improvement': 0.5,
    'blunder_avoided': 0.3,
    'continued_play': 0.2,
}

# =============================================================================
# Data Pipeline Parameters
# =============================================================================
POSITIONS_PER_GAME = 10
MIN_MOVE_NUMBER = 5
MAX_MOVE_NUMBER = 40
MOVE_SAMPLE_INTERVAL = 5  # Sample every 5th move

# =============================================================================
# Stockfish Configuration
# =============================================================================
STOCKFISH_DEPTH = 15
BLUNDER_THRESHOLD_CP = 100  # centipawns
MISTAKE_THRESHOLD_CP = 50
INACCURACY_THRESHOLD_CP = 20

# =============================================================================
# Student Simulator
# =============================================================================
STUDENT_LEARNING_RATE = 0.01
STUDENT_BASE_CONCEPTS = ['tactics', 'strategy', 'endgame', 'opening', 'calculation']
N_SIMULATED_STUDENTS = 50

# =============================================================================
# Experiment Configuration
# =============================================================================
N_EXPERIMENT_EPISODES = 200
N_INTERACTIONS_PER_EPISODE = 20
RANDOM_STATE = 42

# =============================================================================
# Paths
# =============================================================================
DATA_RAW_DIR = "data/raw/"
DATA_PROCESSED_DIR = "data/processed/"
MODELS_DIR = "models/saved/"
RESULTS_DIR = "results/"
PLOTS_DIR = "results/plots/"
