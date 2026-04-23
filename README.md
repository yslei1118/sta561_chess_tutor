# Chess Tutor: Adaptive Teaching with Probabilistic ML

**STA 561D Final Project** — Yushu Lei, Jiahe Li, Ziyang Qin, Yiming Cui, Zherui Zhang

An adaptive chess tutor that plays moves a human at a target rating would actually
play, picks a teaching style for each moment via a contextual bandit, and
explains positions in rating-appropriate prose rather than engine centipawn
scores.

## Our Pedagogical Stance

A chess *engine* finds the best move. A chess *tutor* wants the student to
improve. We treat those as different objectives and design for the second:

1. **Human-level moves, not optimal moves.** The bot plays what a typical
   player at the target rating would play — including their characteristic
   mistakes — so the student encounters the positions they will see in real
   games, not the positions Stockfish would engineer.
2. **Words, not numbers.** We do not show centipawn evaluations. A student
   reading "+0.3" learns nothing actionable. We replace it with named
   observations ("king safety is shaky, trade the queens") calibrated to the
   student's rating.
3. **Varied feedback, not a single voice.** A blunder warning is not a
   strategy lesson and a strategy lesson is not encouragement. We decompose
   teaching into seven distinct intents and pick the one that fits the moment
   — which is itself a learned decision.
4. **Honest evaluation.** We verify that our teaching system is using actual
   pedagogical signal (not recovering a rule we wrote into its own reward)
   via a concept-map-shuffling sanity check.

## What the system does

Two entry points:

- **Position Evaluator (Python API / CLI).**
  `from chess_tutor.interactive import evaluate_fen, format_report` — pass a
  FEN and a target ELO, get an assessment, a plan, the top human-likely moves
  at that rating, and all seven feedback styles side-by-side.
- **Play against the Bot (notebook Section 10).** Click-to-move SVG board.
  The bot plays at your chosen rating. After each move the tutor produces a
  commentary (piece-specific, tier-adapted language) and a
  Thompson-Sampling-selected feedback.

Under the hood:

- **Three move-prediction architectures** trained on ~600K Lichess positions
  across five rating brackets (1100 / 1300 / 1500 / 1700 / 1900): per-bracket
  Random Forest (A), pooled RF with rating as a feature (B), and
  Nadaraya–Watson kernel interpolation across brackets (C) for continuous
  ratings like ELO 1640.
- **35,140 Stockfish depth-12 labels** on real played moves, used as the
  simulator's cp_loss distribution (so the "mistakes the simulated student
  makes" are calibrated to real-world distributions, not invented).
- **Seven feedback types × three ELO tiers** of language templates, filled
  from live position features.
- **Five bandit policies** (Thompson Sampling, LinUCB, ε-Greedy, Rule-Based,
  Random) compared across 1,000 simulated sessions × 50 moves.
- **ZPD-based student simulator** that updates concept-by-concept mastery
  from the feedback style the bandit picks, which in turn shifts the next
  turn's cp_loss percentile draw.

## Installation

Python 3.11+ recommended.

```bash
git clone <repo>
cd sta561_chess_tutor
conda create -n chess_tutor python=3.11 -y
conda activate chess_tutor
pip install -r requirements.txt
```

Install Stockfish if you want to regenerate the cp_loss labels used in the
simulator:

```bash
brew install stockfish        # macOS
sudo apt install stockfish    # Debian/Ubuntu
# Windows: https://stockfishchess.org/download/
```

## Build the dataset and train the models

Data files are not checked in (~330 MB). Reproduce them from the public
Lichess January-2014 dump:

```bash
mkdir -p data/raw data/processed models/saved results/plots

# 1. Download + decompress the PGN (~550 MB decompressed, ~1 min on good network)
python -c "from chess_tutor.data.download import download_lichess_pgn; \
           download_lichess_pgn(year=2014, month=1)"

# 2. Parse → one row per sampled position (~15 min, single-threaded)
python scripts/parse_positions.py

# 3. Build candidate-ranking arrays (~3 min)
python scripts/build_candidate_dataset.py

# 4. Label cp_loss with Stockfish depth 12 (required for §12 bandit numbers;
#    train_and_evaluate does NOT consume these labels — skip if you only want
#    to run the bot / demo notebook; ~15 min on an M1 Pro with 8 workers)
python scripts/label_real_blunders.py

# 5. Train Architectures A, B, C (~8 min on M1 Pro / 4-core laptop)
python scripts/train_and_evaluate.py
```

After the trained models land in `models/saved/`, the full experiment scripts
become runnable. The bot still has heuristic fallbacks when trained models are
absent, but the reported results use the trained artifacts above.

## How to run

- **Full demo notebook:** `Kernel → Restart & Run All` on
  `chess_tutor/demo/chess_tutor_demo.ipynb`. Sections 1–8 are static; 9–10 are
  interactive.
- **Reproduce the main bandit experiment:**
  `python scripts/run_final_experiment.py` (writes
  `results/bandit_comparison.csv` and several plots in `results/plots/`).
- **Reproduce the hyperparameter sweep:**
  `python scripts/bandit_hyperparam_sweep.py`.
- **Reproduce the architecture ablation:**
  `python scripts/arch_c_retune.py`.
- **Reproduce the Stockfish labels:**
  `python scripts/label_real_blunders.py`. Defaults to `stockfish` on `PATH`;
  override with `STOCKFISH_PATH=/abs/path/to/stockfish`.
- **Run the test suite:** `pytest tests/ -q` (174 tests, ~3 seconds).

## Repository layout

```
sta561_chess_tutor/
├── chess_tutor/                  # main package (~4,400 LOC)
│   ├── bot/
│   │   ├── player.py             # ChessTutorBot with ELO-conditioned move scoring
│   │   └── commentary.py         # tier-adaptive move commentary
│   ├── feedback/                 # 7 feedback types × 3 ELO tiers of templates
│   ├── models/
│   │   ├── move_predictor.py     # A/B/C architectures
│   │   ├── kernel_interpolation.py  # Nadaraya-Watson across ELO brackets
│   │   └── candidate_predictor.py   # wrapper over trained RF artifacts
│   ├── teaching/
│   │   ├── bandit.py             # 5 policies (TS, LinUCB, ε-Greedy, Rule, Random)
│   │   ├── context.py            # 20-D bandit context builder
│   │   └── reward.py             # analytic reward formula
│   ├── simulation/
│   │   ├── student_simulator.py  # ZPD learning model
│   │   └── runner.py             # episode runner + sanity-check harness
│   └── demo/chess_tutor_demo.ipynb   # interactive demo
├── scripts/                      # training, labeling, sweeps, ablation
├── tests/                        # 174 pytest tests
├── data/                        # not tracked — regenerate via the commands above
│   ├── raw/                     # Lichess PGN dump
│   └── processed/               # parquet + numpy arrays consumed by training
├── results/                      # CSVs and plots
├── README.md                     # this file
├── EXECUTIVE_SUMMARY.md          # 2-page non-technical overview
├── FAQ.md                        # skeptical-reader Q&A
└── TECHNICAL_APPENDIX.md         # full technical report
```

## Where to read what

- **30-second overview:** this README.
- **2-page non-technical summary:** `EXECUTIVE_SUMMARY.md`.
- **Skeptical-reader Q&A** (why a bandit, why Maia not Stockfish, why the
  simulator is not circular, …): `FAQ.md`.
- **Full technical report** with all experimental details, tables, and plots:
  `TECHNICAL_APPENDIX.md`.

## Key results (from committed CSVs)

- **Move prediction top-1 per bracket:** ~16–17% (vs ~3% random over ~30
  legal moves); top-5 ~42%.
- **Stockfish-labeled blunder rate across rating brackets:** ~10%, roughly
  stable from 1100 to 1900 (a real finding about how Lichess matchmaking
  distributes difficulty).
- **Bandit comparison (1,000 sessions × 50 moves, empirical reward + 0.25
  alignment bonus):** the three bandit policies (TS 43.57, ε-Greedy 43.72,
  LinUCB 43.24) all beat Random (42.67) and Rule-Based (40.90) on mean
  cumulative reward, but cluster within a std of ~5.7 and are
  statistically indistinguishable from each other. The gap is sharper on
  mean student ELO gain: ε-Greedy 352.7, TS 285.9, LinUCB 254.8, Random
  248.8, Rule-Based 148.1 — bandits lift simulated students ~70–140% above
  the rule-based baseline. Full table in `results/bandit_comparison.csv`.
- **Self-reference sanity check:** shuffling the (feedback type → chess
  concept) mapping inside the simulator drops bandit performance to
  Random-baseline level, confirming the bandit is using the real mapping
  rather than recovering a pre-committed reward pattern.

## Known limitations

- The simulated student is an assumption, not a real learner. We report
  policy *rankings* (robust across calibrations) rather than absolute rating
  gains (calibration-dependent).
- Our per-bracket move-prediction top-1 of 16–17% lags Maia-scale neural
  models trained on millions of games; we deliberately chose a smaller RF
  stack that trains on CPU in minutes and ships a transparent feature set.
- Tactical concepts (tactics / strategy / endgame) are hand-mapped to
  feedback types. We document this as an explicit assumption and added the
  sanity check above precisely to prevent circular reasoning about it.
- Notebook Sections 9 and 10 are `ipywidgets` interactive UIs — the static
  GitHub preview will not render them. Open the notebook locally.

## Deliverables

- `README.md` — this file.
- `EXECUTIVE_SUMMARY.md` — 2-page non-technical overview.
- `FAQ.md` — skeptical-reader Q&A.
- `TECHNICAL_APPENDIX.md` — full technical report.
- `chess_tutor/demo/chess_tutor_demo.ipynb` — runnable notebook demo.
- `results/` — all CSVs and figures.
- `tests/` — 174 pytest tests.
