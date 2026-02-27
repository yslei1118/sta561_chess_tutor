# Chess Tutor: Master Roadmap (v2 — Literature-Informed)

> Updated to reflect 90+ paper lit review. Each phase now references specific papers and proven techniques.

---

## Phase 1: Data Pipeline & Feature Engineering

**Timeline:** Week 1–2  
**STA 561 Topics:** None (infrastructure)  
**Key Reference:** Maia-1 (KDD 2020) data methodology

### Goals
- Download 2–3 months of Lichess rated standard games
- Parse PGN into per-position records with player ELO
- Extract ~30 handcrafted board-state features + ~10 move-context features
- Create stratified train/test splits by ELO bracket (1100, 1300, 1500, 1700, 1900)
- Pre-load Stockfish evaluations from HuggingFace (avoid running engine yourself)

### Data Sources
| Source | What | Size | Format |
|--------|------|------|--------|
| database.lichess.org | All rated games (PGN monthly files) | ~20-50 GB/month compressed | .pgn.zst |
| HuggingFace Lichess/chess-position-evaluations | 342M positions with Stockfish evals | ~50GB | parquet |
| HuggingFace Lichess/chess-puzzles | 5.75M puzzles with Glicko-2 ratings + themes | ~1GB | csv |
| HuggingFace Lichess/standard-chess-games | Streaming PGN loader | variable | parquet |

### Feature Set (from BP-Chess validation: handcrafted features + simple ML ≥ Maia accuracy)

**Board-State Features (per position, ~30 dims):**
- Material count per piece type per side (12)
- Material balance in centipawns (1)
- Piece mobility: legal move count for side to move (1)
- King safety: pawn shield integrity, attacks on king zone (2)
- Center control: pieces/pawns on e4,d4,e5,d5 (1)
- Pawn structure: isolated, doubled, passed pawns, pawn islands (4)
- Development: minor pieces off starting squares (1)
- Castling rights: 4 binary (4)
- Game phase: opening/middlegame/endgame by material (1 categorical → 3 one-hot)
- Threat count: pieces under attack, hanging pieces (2)

**Move-Context Features (per candidate move, ~10 dims):**
- Is capture (1 binary)
- Is check (1 binary)
- Piece type being moved (6 one-hot)
- Stockfish eval delta from best move (1 continuous)
- Is book move (1 binary)
- Move number normalized (1 continuous)

### Sampling Strategy (following Maia-1)
- Sample 100K–500K games per ELO bracket
- ELO brackets: 1100±50, 1300±50, 1500±50, 1700±50, 1900±50
- Time controls: standard and rapid only (no bullet/blitz — too much time pressure noise)
- Train/test split: temporal (train on months 1-2, test on month 3)
- Extract ~10 positions per game (every 5th move after move 5, until move 40)

### Deliverables
- `data/download.py` — fetch + decompress Lichess PGN
- `data/parse_pgn.py` — PGN → DataFrame of (game_id, move_num, fen, white_elo, black_elo, move_uci, result)
- `data/extract_features.py` — FEN → 30-dim feature vector
- `data/stockfish_eval.py` — Stockfish eval wrapper (or HuggingFace lookup)
- `data/dataset.py` — ChessTutorDataset class with load/save/split/filter methods
- Output: `data/processed/` directory with parquet files per ELO bracket

---

## Phase 2: ELO-Conditioned Move Prediction

**Timeline:** Week 2–5  
**STA 561 Topics:** Classification (T4), Kernel Methods (T3), Regularization (T2), Random Forests (T7)  
**Key References:** Maia-1 (KDD 2020), Maia-2 (NeurIPS 2024), BP-Chess (arXiv 2504.05425), Nadaraya-Watson (1964)

### Problem Formulation
Given board features **x** and player skill level **s** (ELO), predict the move **m** from legal moves **M(x)**:
```
P(m | x, s) for m ∈ M(x)
```

### Three Architectures (Compare All)

**Model A: Separate-by-ELO (Maia-1 style)**
- Train one RF/SVM classifier per ELO bracket
- No interpolation between brackets
- Baseline approach; simple and interpretable
- *Paper support:* Maia-1 achieves ~51% with separate neural nets; BP-Chess achieves 25% improvement with handcrafted features + LinearSVC

**Model B: ELO-as-Feature (Pooled)**
- Append normalized ELO as extra feature
- Single model on all data
- Tests whether one model can capture skill-dependent behavior
- *Paper support:* Maia-2 validates unified models outperform separate ones

**Model C: Kernel-Smoothed Interpolation (NOVEL — primary contribution)**
- Train separate per-bracket models (like Model A)
- At inference, combine predictions using Nadaraya-Watson kernel regression:
  ```
  P(m|x, s*) = Σ_k K_h(s* - s_k) · P_k(m|x) / Σ_k K_h(s* - s_k)
  ```
  where K_h is Gaussian kernel with bandwidth h, s_k are training ELO centers
- Enables continuous ELO interpolation without retraining
- Bandwidth h selected by cross-validation (leave-one-bracket-out)
- *Paper support:* Nadaraya (1964), Watson (1964). Kossen et al. (2022) show NW on learned features gives excellent calibration. Sarkar & Cooper (AIIDE 2017) demonstrate GP regression for game-based skill estimation.
- *STA 561 connection:* Kernel Methods (T3), Regularization (T2) for bandwidth selection

### Move Encoding Strategy
- **Top-K candidate moves:** For each position, generate top 10 legal moves by Stockfish eval
- **Binary classification per candidate:** P(human_played = 1 | features, candidate_move_features, ELO)
- **Alternative:** Multi-class over top-K candidates with softmax normalization
- Select encoding that gives best move-matching accuracy

### Key Metrics
| Metric | Target | Method |
|--------|--------|--------|
| Move-matching accuracy (top-1) | >35% (RF), >45% (best model) | Fraction of test positions where predicted move = actual human move |
| Move-matching accuracy (top-3) | >55% | Actual human move in top 3 predictions |
| Accuracy monotonicity | Accuracy increases when model ELO matches test ELO | Cross-ELO evaluation matrix |
| Centipawn loss correlation | r > 0.3 | Predicted move quality correlates with actual move quality |

### Ablation Studies
1. Model A vs B vs C
2. Feature subsets: material-only → +mobility → +king_safety → full
3. Classifier comparison: RF vs SVM(RBF) vs LogisticRegression vs Ridge
4. Kernel bandwidth sweep for Model C
5. Training data size: 10K → 50K → 100K → 500K positions per bracket

### Deliverables
- `models/move_predictor.py` — MovePredictor class with fit/predict for all 3 architectures
- `models/kernel_interpolation.py` — NadarayaWatsonELO class
- Cross-ELO evaluation matrix (heatmap)
- Feature importance plots (RF Gini, SVM coefficient magnitudes)

---

## Phase 3: Position Evaluation & Feedback Generation

**Timeline:** Week 4–6  
**STA 561 Topics:** Regression (T1), Random Forests (T7), SVMs (T5), Partial Linear Models  
**Key References:** McGrath et al. (PNAS 2022) concept extraction, Anderson et al. (KDD 2016) human error, Sadikov et al. (CG 2006) automated commentary, BP-Chess blunder prediction

### 3.1 Blunder Detection
Binary classification: given a position + move, predict whether the move loses >100cp (blunder).

**Features:** Board-state features + move-context features + Stockfish eval before and after
**Models:** RF, Gradient Boosted Trees, Logistic Regression  
**Target AUC:** >0.75  
**Paper support:** BP-Chess blunder prediction achieves 0.801 AUC with DeepFM-inspired architecture; we target simpler models for interpretability

### 3.2 Position Complexity Estimation
Regression: predict how "difficult" a position is for a player at a given ELO.

**Definition of difficulty:** Standard deviation of centipawn loss across moves by players at that ELO bracket  
**Features:** Board-state features  
**Models:** Ridge Regression, Random Forest Regression  
**Paper support:** Anderson et al. (KDD 2016) showed position difficulty is stronger predictor than player skill; De Marzo & Servedio (Nature Sci Reports 2023) quantify opening complexity

### 3.3 Feedback Taxonomy (7 types)

| ID | Type | When to Trigger | Example |
|----|------|----------------|---------|
| F1 | Tactical Alert | Tactic available (fork, pin, skewer, etc.) | "There's a knight fork opportunity on c7!" |
| F2 | Strategic Nudge | Positional advantage available | "Your pieces are undeveloped — consider Nf3 to control the center." |
| F3 | Blunder Warning | High blunder probability for candidate move | "Be careful — Bxf7 looks tempting but loses your bishop for a pawn." |
| F4 | Pattern Recognition | Position matches a known pattern | "This is a classic minority attack position in the Queen's Gambit." |
| F5 | Move Comparison | Compare student move with engine recommendation | "You played e5 (decent). The engine prefers d5 because it opens the c-file." |
| F6 | Encouragement | Student made a good move at their level | "Nice move! That pawn push was the engine's top choice." |
| F7 | Simplification | Position is complex; suggest simplifying | "This position is very sharp. Consider trading queens to simplify." |

**Feedback selection** feeds into Phase 4 (contextual bandit decides which type to show).

### 3.4 ELO-Adaptive Feedback
- Same position, different feedback for different ELOs
- 1200 player: "Don't hang your knight!" (F3)
- 1800 player: "Consider Nf5 to exploit the weak d6 pawn" (F2)
- Implementation: feedback generator receives (position, student_elo, feedback_type) → text
- **Paper support:** Kochmar et al. (AIED 2020) showed personalized feedback type selection significantly improves learning outcomes

### Deliverables
- `models/position_eval.py` — BlunderDetector class, PositionComplexity class
- `feedback/taxonomy.py` — FeedbackType enum
- `feedback/templates.py` — Template strings with slot-filling
- `feedback/generator.py` — FeedbackGenerator class: (position, elo, feedback_type) → text

---

## Phase 4: Adaptive Teaching Engine (CORE INNOVATION)

**Timeline:** Week 5–8  
**STA 561 Topics:** Contextual Bandits (T14), Active Learning (T9), Bayesian Methods  
**Key References:** Agrawal & Goyal (ICML 2013), Russo et al. (Tutorial 2018), Lan & Baraniuk (EDM 2016), De Kerpel et al. (arXiv 2602.04347, 2025), Clement et al. (JEDM 2015)

### 4.1 Contextual Bandit Formulation

**Context vector x_t** (built from position + student state, ~20 dims):
- Board-state features (subset): material_balance, mobility, king_safety, game_phase (4)
- Position complexity (from Phase 3) (1)
- Blunder probability (from Phase 3) (1)
- Student ELO (normalized) (1)
- Student recent performance: avg centipawn loss over last 5 moves (1)
- Student weakness profile: tactical_score, strategic_score, endgame_score (3)
- Move number / game progress (1)
- Whether student is in time pressure (1)
- Running blunder count this game (1)
- Student trend: improving / stable / declining over session (3 one-hot)
- Game result probability from current position (1)

**Arms a ∈ {F1, ..., F7}:** The 7 feedback types from Phase 3

**Reward r_t:** Composite signal measuring teaching effectiveness:
- Primary: did the student's next move improve? (centipawn loss decreased)
- Secondary: did the student avoid a blunder they were likely to make?
- Tertiary: student engagement proxy (continued playing / didn't quit)
- Combined: r_t = α · (cp_improvement) + β · (blunder_avoided) + γ · (continued_play)
  where α, β, γ are tunable weights

### 4.2 Algorithm: Thompson Sampling for Linear Contextual Bandits

Following Agrawal & Goyal (ICML 2013):

```
For each arm a ∈ {1,...,K}:
  Maintain: B_a = I_d + Σ x_t x_t^T    (d×d precision matrix)
            f_a = Σ r_t x_t              (d-dim reward accumulator)
            μ̂_a = B_a^{-1} f_a           (posterior mean)

At each round t:
  Observe context x_t
  For each arm a:
    Sample θ̃_a ~ N(μ̂_a, v² B_a^{-1})   (v = exploration parameter)
    Compute score_a = θ̃_a^T x_t
  Select arm a_t = argmax_a score_a
  Observe reward r_t
  Update B_{a_t} += x_t x_t^T
  Update f_{a_t} += r_t x_t
```

**Exploration parameter v:** Start at v=1.0, can tune via offline policy evaluation

**Regret guarantee:** Õ(d^{3/2} √T) where d = context dimension, T = total rounds

### 4.3 Student Simulator (Critical for Evaluation)

Since no live deployment, build a simulated student:

**StudentSimulator(elo, weakness_profile):**
- `weakness_profile`: dict mapping chess concepts to skill levels
  - e.g., {"tactics": 0.6, "strategy": 0.3, "endgame": 0.4, "opening": 0.7}
- `respond_to_position(board, feedback_type, feedback_text)` → move
  - Sample move from Model A/C at the student's ELO
  - If feedback was relevant to position (e.g., tactical alert when tactic exists), improve move quality with probability proportional to:
    - p_learn = base_rate × relevance(feedback_type, position) × (1 - mastery(concept))
  - This models Zone of Proximal Development: feedback helps most when student is near but not at mastery
- `update_state(move_quality, feedback)` → update weakness_profile via Bayesian update

**Paper support:** Clement et al. (JEDM 2015) ZPDES algorithm; Rafferty et al. (AIED 2011) teaching as POMDP; Chounta et al. (EDM 2017) "Grey Area" concept

### 4.4 Baseline Policies (Must Compare Against)
1. **Random:** Select feedback type uniformly at random
2. **Always-Tactical:** Always show tactical feedback (F1)
3. **Always-Strategic:** Always show strategic feedback (F2)
4. **Rule-Based Heuristic:** If blunder_prob > 0.5 → F3; if tactic available → F1; else F2
5. **ε-Greedy:** ε=0.1 with empirical mean rewards
6. **LinUCB:** UCB variant from Li et al. (WWW 2010)

### 4.5 Evaluation Metrics
| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Cumulative regret | Learning efficiency | Sub-linear growth |
| Simulated ELO gain after N episodes | Teaching effectiveness | 2× faster than random |
| Feedback diversity | Exploration quality | All 7 types used, no arm starved |
| Arm selection entropy | Policy richness | H > 1.5 bits |
| Reward convergence | Algorithm stability | Converges within 500 episodes |

### Deliverables
- `teaching/bandit.py` — LinearThompsonSampling class
- `teaching/context_builder.py` — build_context(board, student_state) → vector
- `teaching/reward.py` — compute_reward(move_quality_before, move_quality_after, ...)
- `simulation/student_simulator.py` — StudentSimulator class
- `simulation/episode_runner.py` — run_episode(student, tutor, n_positions) → trajectory
- Regret curves, comparison plots

---

## Phase 5: Interactive System & Demo

**Timeline:** Week 7–10  
**STA 561 Topics:** RL/MDP (T15)  
**Key References:** KL-Regularized Search (ICML 2022), Maia-1 bot interface

### 5.1 Position Evaluator (A+ Requirement 1)
**Input:** FEN string + target ELO  
**Output:** ELO-appropriate position evaluation with feedback

Pipeline: FEN → extract_features → move_predictor(elo) → position_eval → feedback_generator → display

### 5.2 Playing Bot (A+ Requirement 2)
**Behavior:** Play at a target ELO level with running commentary

**Move selection:** Sample from P(m|x, target_elo) using Model C (kernel interpolation)
- Optional: KL-regularized blend between human model and Stockfish
  ```
  π_bot(m|x) ∝ π_human(m|x,elo) · exp(λ · Q_engine(x,m))
  ```
  where λ controls strength (λ=0 → pure human, λ→∞ → pure engine)

**Commentary generation:** After each move, generate contextual commentary:
- What the bot played and why (from bot's "perspective" at that ELO)
- What the student played and whether it was good/bad
- Teaching suggestion for the student

### 5.3 Jupyter Demo Notebook
- Cell 1: Setup (imports, load models)
- Cell 2: Position Evaluator — input FEN + ELO via widgets, see evaluation + feedback
- Cell 3: Play Against Bot — interactive game with commentary
- Cell 4: Teaching Session — play through a game with adaptive feedback
- Cell 5: Visualization — show ELO interpolation curves, bandit learning curves

**UI:** ipywidgets for ELO slider, FEN input, move input. python-chess SVG for board rendering.

### Deliverables
- `bot/engine.py` — ChessTutorBot class with play_move(board, target_elo)
- `bot/commentary.py` — CommentaryGenerator class
- `demo/tutor_demo.ipynb` — Main interactive notebook
- `demo/analysis_notebook.ipynb` — Results and ablation visualization

---

## Phase 6: Write-Up & Final Submission

**Timeline:** Week 10–12

### Deliverables
1. **Executive Summary (2 pages, no math):** Problem → Approach → Results → Future Work
2. **FAQ (2–5 pages):** 7–10 questions an intelligent skeptic would ask
3. **Technical Appendix:** All math, algorithms, code, results, reproducibility instructions

See `07_DELIVERABLES_GUIDE.md` for detailed templates.

---

## Week-by-Week Timeline

| Week | Phase | Milestones | Risks |
|------|-------|-----------|-------|
| 1 | P1 | Download data, python-chess setup, extract features from 1K games | PGN parsing speed |
| 2 | P1→P2 | Full pipeline on 100K+ games; first RF on single bracket | Feature engineering decisions |
| 3 | P2 | All 3 architectures implemented; initial accuracy | Move encoding choice |
| 4 | P2→P3 | Cross-ELO evaluation complete; blunder detector started | Kernel bandwidth tuning |
| 5 | P2+P3 | Phase 2 ablations done; feedback taxonomy implemented | Balancing depth vs breadth |
| 6 | P3→P4 | Feedback generator working; start Thompson Sampling | Reward function design |
| 7 | P4 | Bandit + simulator running; initial regret curves | Simulation fidelity |
| 8 | P4→P5 | Bandit vs baselines complete; start bot | Integration complexity |
| 9 | P5 | Position evaluator + bot + commentary working | Widget debugging |
| 10 | P5 | Demo notebook polished; qualitative testing | Recruiting testers |
| 11 | P6 | Executive summary + FAQ drafts | Writing quality |
| 12 | P6 | Technical appendix; final polish; submit | Reproducibility check |

---

## Differentiation from Existing Work

| Dimension | Maia / ALLIE | This Project |
|-----------|-------------|-------------|
| **Goal** | Predict human moves | Maximize learner improvement speed |
| **Methods** | Deep learning (AlphaZero arch) | Statistical ML (RF, SVM, kernels) + Contextual Bandits |
| **Teaching** | Not addressed | Formalized as bandit problem with algorithm + evaluation |
| **ELO modeling** | Separate NNs (Maia-1) or Transformer (Maia-2) | Kernel regression interpolation (interpretable) |
| **Evaluation** | Move-matching accuracy | Teaching effectiveness (simulated ELO improvement) |
| **Novelty** | Architecture engineering | Novel formulation: chess features as bandit context for feedback selection |
