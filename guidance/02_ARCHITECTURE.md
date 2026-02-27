# Chess Tutor: System Architecture

## System Data Flow

```
                            ┌─────────────────────────────────────┐
                            │         PHASE 1: DATA PIPELINE       │
                            │                                       │
  Lichess PGN ──→ parse_pgn ──→ extract_features ──→ dataset.py   │
  HF Evals   ──→ stockfish_eval ─────────────────→ (parquet files)│
                            └──────────────┬────────────────────────┘
                                           │
                                     features + labels
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
          ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
          │ PHASE 2: MOVE   │   │ PHASE 3: POSITION│   │ PHASE 3: BLUNDER │
          │ PREDICTION      │   │ COMPLEXITY       │   │ DETECTION        │
          │                 │   │                  │   │                  │
          │ Model A (per-ELO│   │ Ridge/RF regress │   │ RF/GBT binary    │
          │ Model B (pooled)│   │ → difficulty score│   │ → blunder_prob   │
          │ Model C (kernel)│   │                  │   │                  │
          └────────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
                   │                     │                       │
                   │         ┌───────────┴───────────────────────┘
                   │         │
                   ▼         ▼
          ┌──────────────────────────────────────────────────────┐
          │              PHASE 4: TEACHING ENGINE                 │
          │                                                       │
          │  context_builder ──→ [board features, student state,  │
          │                       complexity, blunder_prob]       │
          │         │                                             │
          │         ▼                                             │
          │  bandit.py (Thompson Sampling)                        │
          │         │                                             │
          │         ▼                                             │
          │  feedback_type ──→ generator.py ──→ feedback_text     │
          │                                                       │
          │  reward.py ◄── student response (via simulator)       │
          └──────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │              PHASE 5: INTERACTIVE DEMO                │
          │                                                       │
          │  Position Evaluator: FEN + ELO → evaluation + feedback│
          │  Playing Bot: sample moves from P(m|x, target_elo)    │
          │  Commentary: real-time move analysis at student level  │
          │  Jupyter widgets: ipywidgets + python-chess SVG        │
          └──────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
config.py ◄──── (everything imports this)
    │
    ├── data/
    │   ├── download.py       (standalone, no internal deps)
    │   ├── parse_pgn.py      ◄── utils/chess_helpers.py
    │   ├── extract_features.py ◄── utils/chess_helpers.py, config.py
    │   ├── stockfish_eval.py (standalone, wraps python-chess engine)
    │   └── dataset.py        ◄── parse_pgn, extract_features, stockfish_eval
    │
    ├── models/
    │   ├── move_predictor.py  ◄── data/dataset.py, config.py
    │   ├── kernel_interpolation.py ◄── move_predictor.py
    │   └── position_eval.py   ◄── data/dataset.py, config.py
    │
    ├── feedback/
    │   ├── taxonomy.py        (standalone, defines enums)
    │   ├── templates.py       ◄── taxonomy.py
    │   └── generator.py       ◄── taxonomy, templates, models/position_eval
    │
    ├── student/
    │   ├── model.py           ◄── config.py
    │   └── tracker.py         ◄── model.py
    │
    ├── teaching/
    │   ├── context_builder.py ◄── models/position_eval, student/model
    │   ├── reward.py          ◄── config.py
    │   └── bandit.py          ◄── context_builder, reward
    │
    ├── simulation/
    │   ├── student_simulator.py ◄── student/model, models/move_predictor
    │   └── episode_runner.py    ◄── student_simulator, teaching/bandit, feedback/generator
    │
    ├── bot/
    │   ├── engine.py          ◄── models/move_predictor, models/kernel_interpolation
    │   └── commentary.py      ◄── feedback/generator, models/position_eval
    │
    ├── evaluation/
    │   ├── metrics.py         (standalone utility functions)
    │   ├── ablation.py        ◄── models/*, evaluation/metrics
    │   └── offline_policy_eval.py ◄── simulation/episode_runner, metrics
    │
    └── utils/
        ├── chess_helpers.py   (standalone, wraps python-chess)
        └── visualization.py   ◄── utils/chess_helpers
```

## Key Design Decisions

### 1. Feature-Based (Not Neural)
We use handcrafted interpretable features rather than raw board tensors because:
- STA 561 is a statistical learning course → RF, SVM, kernel methods are the focus
- BP-Chess (2025) proved handcrafted features + simple ML matches/beats Maia
- Enables feature importance analysis, partial dependence plots, interpretable models
- Much faster training (minutes, not hours/days)

### 2. Modular .py Architecture (Not Monolithic Notebook)
- Each module is independently testable and importable
- Demo notebooks are thin wrappers that import from modules
- Enables unit testing, version control, and collaboration
- Professor can run individual modules to verify functionality

### 3. Kernel Interpolation as Novel Contribution
- Nadaraya-Watson kernel regression over ELO is our theoretical innovation
- Connects directly to STA 561 kernel methods topic
- More interpretable than Maia-2's Transformer-based skill embedding
- Allows continuous ELO queries without retraining

### 4. Thompson Sampling (Not UCB or ε-Greedy)
- Natural Bayesian framework aligns with course philosophy
- Posterior uncertainty drives exploration (principled, not ad hoc)
- Strong theoretical guarantees (Agrawal & Goyal 2013)
- Proven in educational settings (De Kerpel et al. 2025, Clement et al. 2015)

### 5. Simulator-Based Evaluation
- No live deployment possible within course timeframe
- Student simulator enables controlled, reproducible experiments
- Offline policy evaluation (IPS, DR) for comparing policies without new data
- Supplement with qualitative testing from 3-5 real players

## config.py Constants

```python
# ELO brackets
ELO_BRACKETS = [1100, 1300, 1500, 1700, 1900]
ELO_BRACKET_WIDTH = 50  # ±50 around center

# Feature dimensions
N_BOARD_FEATURES = 30
N_MOVE_FEATURES = 10
N_CONTEXT_FEATURES = 20  # for bandit

# Feedback types
N_FEEDBACK_TYPES = 7

# Model hyperparameters
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 20
SVM_KERNEL = 'rbf'
SVM_C = 1.0
KERNEL_BANDWIDTH_DEFAULT = 100.0  # ELO units

# Bandit hyperparameters
TS_EXPLORATION_V = 1.0
TS_PRIOR_VARIANCE = 1.0
REWARD_WEIGHTS = {'cp_improvement': 0.5, 'blunder_avoided': 0.3, 'continued_play': 0.2}

# Data parameters
POSITIONS_PER_GAME = 10
MIN_MOVE_NUMBER = 5
MAX_MOVE_NUMBER = 40
MOVE_SAMPLE_INTERVAL = 5

# Stockfish
STOCKFISH_DEPTH = 15
BLUNDER_THRESHOLD_CP = 100  # centipawns
```

## Data Schemas

### PositionRecord (one row per sampled position)
```
game_id:        str         # Lichess game ID
move_number:    int         # Move number in game
fen:            str         # FEN string of position before move
white_elo:      int         # White player's ELO
black_elo:      int         # Black player's ELO
side_to_move:   str         # 'white' or 'black'
player_elo:     int         # ELO of player to move
move_uci:       str         # Move played in UCI format (e.g., 'e2e4')
move_san:       str         # Move in SAN format (e.g., 'e4')
result:         str         # '1-0', '0-1', '1/2-1/2'
time_control:   str         # e.g., '600+0'
features:       float[30]   # Board-state feature vector
move_features:  float[10]   # Move-context feature vector
stockfish_eval: float       # Stockfish eval in centipawns (positive = white advantage)
best_move_uci:  str         # Stockfish best move
cp_loss:        float       # Centipawn loss from best move (always ≥ 0)
is_blunder:     bool        # cp_loss > BLUNDER_THRESHOLD_CP
```

### StudentState
```
elo:                 int          # Current ELO estimate
weakness_profile:    dict[str, float]  # concept → skill level [0, 1]
recent_cp_losses:    list[float]  # Last N centipawn losses
blunder_count:       int          # Blunders in current session
moves_played:        int          # Total moves in session
trend:               str          # 'improving', 'stable', 'declining'
```

### TeachingEpisode (one row per teaching interaction)
```
episode_id:     int
position_fen:   str
student_elo:    int
context_vector: float[20]
feedback_type:  int          # 0–6 (F1–F7)
feedback_text:  str
student_move:   str          # Move played after feedback
move_quality:   float        # Centipawn loss of student's move
reward:         float        # Computed reward
```
