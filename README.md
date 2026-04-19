# Chess Tutor: Adaptive Teaching with Probabilistic ML

> **STA 561D — Probabilistic Machine Learning, Duke University**
>
> An adaptive chess tutoring system that predicts human moves at any ELO level (1100–1900), provides skill-appropriate feedback, and uses contextual Thompson Sampling to select optimal teaching strategies.

## Key Ideas

1. **ELO-Conditioned Move Prediction** — Given a board position and a target ELO, predict which move a human at that skill level would play. Three architectures compared: per-bracket models (A), pooled with ELO feature (B), and Nadaraya-Watson kernel interpolation across brackets (C).

2. **Kernel Interpolation (Methodological Extension)** — Gaussian kernel smoothing over ELO brackets enables **continuous skill-level queries** without retraining (e.g., ELO=1640 between training brackets). Bandwidth selected via leave-one-bracket-out CV. On raw top-1 accuracy, the simpler pooled model (B) actually wins; C's contribution is inference-time flexibility, not accuracy.

3. **Contextual Thompson Sampling** — A linear contextual bandit selects among 7 feedback types (tactical alert, blunder warning, encouragement, etc.) based on a 20-dimensional context vector capturing board state, student skill, and position complexity.

4. **Student Simulator** — A ZPD-based learning model simulates student improvement for offline policy evaluation: `p_learn = lr × relevance × (1 - mastery)`.

## Results

| Metric | Value |
|--------|-------|
| Move prediction top-1 accuracy per bracket (Arch A, RF) | 13.5%–16.7% (vs ~3% random) |
| Move prediction top-5 accuracy per bracket | 34.7%–42.9% |
| Cross-ELO diagonal dominance | 3/5 brackets (1500/1700/1900); 1100 and 1300 are predicted most accurately by the 1700-bracket model — a real finding about cross-bracket generalization |
| Bandit reward architecture | **pure empirical** — `r = max(0, 1 − cp_loss/200)`, **no alignment term** |
| cp_loss source | empirical percentile draw from 22,712 Stockfish-labeled real moves, bucketed by ELO and modulated by **student's concept-specific mastery** (concept selected by board phase + tactical content) |
| Primary experiment (300 ep × 30 steps, pure empirical reward) | TS 25.65; ε-Greedy 25.65; LinUCB 25.60; Random 25.59; Rule 25.52 (all within 0.5%, std ≈ 4.2) |
| Honest finding | **All 5 policies are statistically indistinguishable under pure empirical reward** — the gap is smaller than the standard deviation. Any ranking between them is within noise. |
| Previous (alignment-based) result | LinUCB 17.96 ≫ others — we removed the alignment term because bandit was mostly recovering our own hand-crafted rule. Under a genuinely empirical reward, the apparent bandit superiority disappears. |
| Regret | Sub-linear ✓ |
| Real Stockfish-labeled blunder rate (across brackets) | ~10% (stable) |

<p align="center">
  <img src="results/plots/cross_elo_heatmap.png" width="45%" />
  <img src="results/plots/regret_curves.png" width="45%" />
</p>
<p align="center">
  <img src="results/plots/kernel_weights.png" width="45%" />
  <img src="results/plots/feature_importance.png" width="45%" />
</p>

## Project Structure

```
sta561_chess_tutor/
├── chess_tutor/                # Main package (35 modules, ~4400 LOC)
│   ├── config.py               # Global constants and hyperparameters
│   ├── data/                   # Data pipeline
│   │   ├── download.py         #   Lichess PGN downloader
│   │   ├── parse_pgn.py        #   PGN → position records
│   │   ├── extract_features.py #   30 board + 10 move features
│   │   ├── stockfish_eval.py   #   Stockfish wrapper
│   │   └── dataset.py          #   ChessTutorDataset class
│   ├── models/                 # ML models
│   │   ├── move_predictor.py   #   Architectures A, B, C
│   │   ├── kernel_interpolation.py  # Nadaraya-Watson ELO kernel
│   │   └── position_eval.py    #   BlunderDetector, PositionComplexity
│   ├── feedback/               # Feedback generation
│   │   ├── taxonomy.py         #   7 feedback types (enum)
│   │   ├── templates.py        #   ELO-adaptive templates
│   │   └── generator.py        #   FeedbackGenerator
│   ├── teaching/               # Adaptive teaching engine
│   │   ├── bandit.py           #   Thompson Sampling + 5 baselines
│   │   ├── context.py          #   20-dim context builder
│   │   └── reward.py           #   Reward computation
│   ├── simulation/             # Offline evaluation
│   │   ├── student_simulator.py #  StudentSimulator + StudentPopulation
│   │   └── runner.py           #   Episode runner + experiment harness
│   ├── bot/                    # Interactive demo
│   │   ├── player.py           #   ChessTutorBot (KL-regularized)
│   │   └── commentary.py       #   Move-by-move commentary
│   ├── evaluation/             # Metrics and ablation
│   │   ├── metrics.py          #   Regret, entropy, AUC, etc.
│   │   └── ablation.py         #   Ablation suite
│   ├── student/model.py        # StudentState dataclass
│   ├── utils/
│   │   ├── helpers.py          #   sigmoid, softmax, normalization
│   │   └── visualization.py    #   10 plot functions
│   └── demo/
│       └── chess_tutor_demo.ipynb  # Interactive demo notebook
├── scripts/                    # Training and evaluation scripts
│   ├── build_candidate_dataset.py
│   ├── train_and_evaluate.py
│   ├── run_phase3_4_5.py
│   └── run_final_experiment.py
├── results/
│   ├── plots/                  # All generated figures
│   ├── ablation_table.csv
│   └── bandit_comparison.csv
├── guidance/                   # Project specification documents
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create conda environment
conda create -n chess_tutor python=3.14 -y
conda activate chess_tutor

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Results

```bash
# 1. Download Lichess data (~93MB decompressed)
python -c "from chess_tutor.data.download import download_lichess_pgn; download_lichess_pgn(2013, 1)"

# 2. Build dataset with features
python -c "
from chess_tutor.data.dataset import ChessTutorDataset
dataset = ChessTutorDataset(data_dir='data/processed/')
dataset.build_from_pgn(['data/raw/lichess_2013_01.pgn'])
dataset.save('chess_tutor_v1')
"

# 3. Build candidate-move dataset for ranking
python scripts/build_candidate_dataset.py

# 4. Train move predictors (Architectures A, B, C)
python scripts/train_and_evaluate.py

# 5. Run bandit experiments and generate all plots
python scripts/run_final_experiment.py

# 6. Launch demo notebook
jupyter notebook chess_tutor/demo/chess_tutor_demo.ipynb
```

## Testing

The project has a comprehensive pytest-based test suite with **140 tests** across
11 modules, covering all core probabilistic ML components: mathematical utilities,
feature extraction, Nadaraya-Watson kernel interpolation, 5 bandit policies,
context builder, reward function, feedback generation, student simulator, and the
ChessTutorBot including its ELO-differentiation behavior. Every test is
deterministic (fixed seed) and fast (<30 s wall-clock).

```bash
# Run all tests
pytest tests/ -q

# With coverage (results written to results/htmlcov/)
pytest tests/ --cov=chess_tutor --cov-report=term-missing --cov-report=html:results/htmlcov -q

# Skip the slow tests (>1s)
pytest tests/ -m "not slow" -q
```

Current coverage on core ML modules:
- `teaching/reward.py`: 100%
- `teaching/bandit.py`: 97%
- `teaching/context.py`: 95%
- `data/extract_features.py`: 96%
- `feedback/generator.py`: 78%
- `simulation/student_simulator.py`: 70%
- `bot/player.py`: 62%
- Data-pipeline modules (`download`, `parse_pgn`, `dataset`, `stockfish_eval`)
  are covered implicitly by the end-to-end notebook run; they are not unit-tested
  because they require network, external binaries, or large input files.

## Methods

### Feature Engineering (30 board + 10 move features)

| Features | Dimensions | Description |
|----------|-----------|-------------|
| Piece counts | 12 | White/Black P, N, B, R, Q, K |
| Material balance | 1 | Centipawn difference |
| Mobility | 1 | Legal move count |
| King safety | 2 | Pawn shield + zone attacks |
| Center control | 1 | Pieces/attacks on central squares |
| Pawn structure | 4 | Isolated, doubled, passed, islands |
| Development | 1 | Minor pieces off starting squares |
| Castling rights | 4 | Binary flags |
| Game phase | 3 | Opening/middlegame/endgame one-hot |
| Hanging pieces | 1 | Undefended pieces under attack |
| Move features | 10 | Capture, check, piece type, cp loss, move number |

### Move Prediction Architectures

- **Architecture A**: Train separate Random Forest per ELO bracket (Maia-1 style)
- **Architecture B**: Single pooled RF with normalized ELO appended as feature
- **Architecture C**: Train per-bracket models, combine via Nadaraya-Watson kernel:

```math
P(m \mid x, s^*) = \frac{\sum_k K_h(s^* - s_k) \cdot P_k(m \mid x)}{\sum_k K_h(s^* - s_k)}
```

where $K_h$ is a Gaussian kernel with bandwidth $h$ selected by leave-one-bracket-out CV.

**Ablation result (real numbers from `results/ablation_table.csv`):**

| Architecture | Top-1 |
|--------------|-------|
| B (Pooled+ELO) | **0.1582** |
| C (Kernel, bw=200) | 0.1533 |
| C (Kernel, bw=300, CV-selected) | see FAQ Q3 — CV picks the largest bandwidth in our sweep; at that width the kernel is near-uniform and C approximates a pooled model |
| A (Per-bracket, RF) | 0.1516 |

B wins on raw top-1. C's value is methodological: it supports **continuous ELO queries** (e.g., ELO=1640) through inference-time kernel smoothing, which B cannot provide meaningfully. See `FAQ.md` Q2–Q3 for the honest discussion.

### Contextual Thompson Sampling

For each feedback arm $a$, maintain Bayesian linear regression $r = \theta_a^\top x + \varepsilon$ with posterior $\mathcal{N}(\hat{\mu}_a, v^2 B_a^{-1})$.

**Select**: Sample $\tilde{\theta}_a \sim \mathcal{N}(\hat{\mu}_a, v^2 B_a^{-1})$, play $\arg\max_a \tilde{\theta}_a^\top x$

**Update**: $B_a \leftarrow B_a + x x^\top$, $f_a \leftarrow f_a + r \cdot x$, $\hat{\mu}_a \leftarrow B_a^{-1} f_a$

### Feedback Types

| # | Type | Target Concepts |
|---|------|----------------|
| F1 | Tactical Alert | Tactics |
| F2 | Strategic Nudge | Strategy, positional play |
| F3 | Blunder Warning | Tactics, calculation |
| F4 | Pattern Recognition | Openings, endgames |
| F5 | Move Comparison | Calculation, strategy |
| F6 | Encouragement | General reinforcement |
| F7 | Simplification | Endgame, strategy |

## Technical Appendix

### Reproducibility

| Item | Detail |
|------|--------|
| Random seed | 42 (consistent across all scripts and models) |
| Python version | 3.14 |
| Key packages | numpy, pandas, scikit-learn, matplotlib, python-chess, pyarrow, zstandard |
| Stockfish | Not required for demo (heuristic fallback available) |

### Data

| Item | Detail |
|------|--------|
| Source | Lichess open database (January 2013) |
| Raw size | ~93 MB decompressed PGN |
| ELO brackets | 1100, 1300, 1500, 1700, 1900 (±50) |
| Sampling | Every 5th move, moves 5–40 |
| Feature dimensions | 30 board + 10 move = 40 |
| Demo dataset | 200 cached positions at `data/demo_cache/` (offline demo) |

### Reward Definition

| Component | Formula |
|-----------|---------|
| **Analytic** (`reward.py`) | `r = 0.5·sigmoid(Δcp) + 0.3·blunder_avoided + 0.2·continued_play` |
| **Simulation** (`runner.py`) | `r = 0.4·base + 0.6·alignment + noise` |

Alignment measures context-arm match (e.g., TACTICAL_ALERT scores higher when position complexity is high). Bandit experiments use simulated reward with known context-arm alignment structure to demonstrate policy learning. This is standard practice for offline bandit evaluation.

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| RF trees | 500 |
| RF max_depth | 20 |
| RF min_samples_leaf | 5 |
| Kernel bandwidth candidates | {25, 50, 75, 100, 150, 200, 300} (leave-one-bracket-out CV) |
| Thompson Sampling v | 1.0 |
| Thompson Sampling prior variance | 1.0 |
| LinUCB α | 1.0 |
| ε-Greedy ε | 0.1 |
| Student simulator learning_rate | 0.01 (ZPD model) |

### Timing (approximate)

| Step | Duration |
|------|----------|
| Data download + parsing | ~5 min |
| Feature extraction | ~10 min |
| Model training (all architectures) | ~15 min |
| Bandit experiments (200 episodes × 5 policies) | ~2 min |
| **Total end-to-end** | **~35 min** |

## References

- McIlroy-Young et al. (2020). *Aligning Superhuman AI with Human Behavior: Chess as a Model System.* KDD 2020. (Maia-1 — Architecture A per-ELO models)
- Skidanov et al. (2025). *BP-Chess: Handcrafted Features for Human Move Prediction.* arXiv:2504.05425. (Feature engineering approach)
- Nadaraya (1964). *On Estimating Regression.* Theory of Probability and Its Applications. (Kernel interpolation — Architecture C)
- Agrawal & Goyal (2013). *Thompson Sampling for Contextual Bandits with Linear Payoffs.* ICML 2013. (Teaching engine)
- Li et al. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010. (LinUCB baseline)
- Clement et al. (2015). *Multi-Armed Bandits for Intelligent Tutoring Systems.* JEDM. (ZPD-based student simulator)
- Jacob et al. (2022). *Modeling Strong and Human-Like Gameplay with KL-Regularized Search.* ICML 2022. (KL-regularized bot)

