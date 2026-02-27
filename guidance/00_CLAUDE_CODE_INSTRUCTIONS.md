# Claude Code: Chess Tutor Project — Master Instructions

> **Read this file first.** It tells you what every other file does and the order to execute.

## Project Summary

Build an adaptive chess tutor for STA 561 (Probabilistic ML) that:
1. Predicts human moves at any ELO (1100–1900) using statistical ML
2. Evaluates positions with ELO-appropriate feedback
3. Selects teaching feedback via contextual Thompson Sampling
4. Provides an interactive demo (position evaluator + playing bot with commentary)

**A+ Requirements:** (a) user inputs FEN → tutor evaluates for a given ELO; (b) user plays against bot with running commentary.

## Document Map

| # | File | Purpose | When to Read |
|---|------|---------|-------------|
| 01 | `MASTER_ROADMAP.md` | High-level phases, timeline, dependencies | Before starting anything |
| 02 | `ARCHITECTURE.md` | System design, module boundaries, data flow | Before writing any code |
| 03 | `API_SPECS.md` | Every function signature, class, data structure | While coding each module |
| 04 | `IMPLEMENTATION_GUIDE.md` | Step-by-step coding instructions per phase | While coding each module |
| 05 | `TESTING_PLAN.md` | Unit tests, integration tests, eval metrics | After each module is coded |
| 06 | `LITERATURE_REFERENCE.md` | Paper summaries, key equations, what to borrow | When implementing algorithms |
| 07 | `DELIVERABLES_GUIDE.md` | How to write the exec summary, FAQ, tech appendix | At the end |

## Execution Order

### Phase 1: Data Pipeline (Week 1–2)
1. Read `02_ARCHITECTURE.md` §1 (project structure)
2. Read `03_API_SPECS.md` §1 (data module interfaces)
3. Read `04_IMPLEMENTATION_GUIDE.md` §1 (step-by-step)
4. Code: `chess_tutor/data/` modules
5. Run: `05_TESTING_PLAN.md` §1 tests
6. **Checkpoint:** 10K+ positions extracted with features, saved as parquet

### Phase 2: ELO-Conditioned Move Prediction (Week 2–5)
1. Read `06_LITERATURE_REFERENCE.md` §1 (Maia, BP-Chess, kernel methods)
2. Read `03_API_SPECS.md` §2 (model interfaces)
3. Read `04_IMPLEMENTATION_GUIDE.md` §2 (three model architectures)
4. Code: `chess_tutor/models/move_predictor.py`
5. Run: `05_TESTING_PLAN.md` §2 tests
6. **Checkpoint:** >35% move-matching accuracy at matched ELO

### Phase 3: Position Evaluation & Feedback (Week 4–6)
1. Read `06_LITERATURE_REFERENCE.md` §3 (blunder detection, feedback taxonomy)
2. Read `03_API_SPECS.md` §3 (evaluator interfaces)
3. Read `04_IMPLEMENTATION_GUIDE.md` §3
4. Code: `chess_tutor/models/position_eval.py`, `chess_tutor/feedback/`
5. Run: `05_TESTING_PLAN.md` §3 tests
6. **Checkpoint:** Blunder detection AUC >0.75; 7 feedback types generating text

### Phase 4: Adaptive Teaching Engine (Week 5–8)
1. Read `06_LITERATURE_REFERENCE.md` §2 (Thompson Sampling, contextual bandits)
2. Read `03_API_SPECS.md` §4 (bandit + simulator interfaces)
3. Read `04_IMPLEMENTATION_GUIDE.md` §4
4. Code: `chess_tutor/teaching/`, `chess_tutor/simulation/`
5. Run: `05_TESTING_PLAN.md` §4 tests
6. **Checkpoint:** Bandit achieves sub-linear regret; outperforms random in simulator

### Phase 5: Interactive System & Demo (Week 7–10)
1. Read `03_API_SPECS.md` §5 (bot + demo interfaces)
2. Read `04_IMPLEMENTATION_GUIDE.md` §5
3. Code: `chess_tutor/bot/`, `chess_tutor/demo/`
4. Run: `05_TESTING_PLAN.md` §5 tests
5. **Checkpoint:** Working Jupyter demo with position eval + bot play + commentary

### Phase 6: Write-Up (Week 10–12)
1. Read `07_DELIVERABLES_GUIDE.md`
2. Generate all plots/tables from `05_TESTING_PLAN.md` §6
3. Write executive summary, FAQ, technical appendix

## Critical Constraints

- **Python only.** All code in `.py` modules, importable and testable.
- **Demo in Jupyter.** 1–2 `.ipynb` notebooks that import from `.py` modules.
- **No deep learning requirement.** RF, SVM, kernel methods are primary. This is a statistics course.
- **Reproducibility required.** Professor must reproduce results from technical description + dataset links.
- **Dependencies:** python-chess, stockfish (engine binary), scikit-learn, numpy, pandas, matplotlib, ipywidgets. Minimize others.
- **STA 561 alignment:** Every model choice must connect to a course topic (listed in each module spec).

## Target Directory Structure

```
chess_tutor/
├── README.md
├── requirements.txt
├── config.py                    # Global constants
├── data/
│   ├── __init__.py
│   ├── download.py              # Download Lichess PGN files
│   ├── parse_pgn.py             # PGN → per-position records
│   ├── extract_features.py      # Board state → feature vector
│   ├── stockfish_eval.py        # Stockfish evaluation wrapper
│   └── dataset.py               # ChessTutorDataset class
├── models/
│   ├── __init__.py
│   ├── move_predictor.py        # 3 architectures for move prediction
│   ├── position_eval.py         # Position evaluation & blunder detection
│   └── kernel_interpolation.py  # Nadaraya-Watson kernel ELO interpolation
├── feedback/
│   ├── __init__.py
│   ├── taxonomy.py              # FeedbackType enum + template registry
│   ├── generator.py             # Generate feedback from position analysis
│   └── templates.py             # NL feedback templates
├── teaching/
│   ├── __init__.py
│   ├── bandit.py                # Linear Thompson Sampling contextual bandit
│   ├── context_builder.py       # Build context vectors
│   └── reward.py                # Reward function definitions
├── student/
│   ├── __init__.py
│   ├── model.py                 # StudentState class
│   └── tracker.py               # ELO update, performance tracking
├── simulation/
│   ├── __init__.py
│   ├── student_simulator.py     # Simulated student
│   └── episode_runner.py        # Run teaching episodes
├── bot/
│   ├── __init__.py
│   ├── engine.py                # ELO-conditioned playing bot
│   └── commentary.py            # Real-time move commentary
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # All evaluation metrics
│   ├── ablation.py              # Ablation study runner
│   └── offline_policy_eval.py   # IPS, DR estimators
├── demo/
│   ├── tutor_demo.ipynb         # Main interactive demo
│   └── analysis_notebook.ipynb  # Results visualization
└── utils/
    ├── __init__.py
    ├── chess_helpers.py          # Board utilities
    └── visualization.py         # Rendering, plotting
```
