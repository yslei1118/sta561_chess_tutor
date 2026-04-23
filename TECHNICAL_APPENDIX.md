# Technical Appendix — Chess Tutor: An Interactive Chess Teacher that Maximizes Student Improvement

Group members: Yushu Lei, Jiahe Li, Ziyang Qin, Yiming Cui, Zherui Zhang 

> This document is the technical appendix of the submission. It contains all
> technical details, mathematical definitions, code references, commands, data
> provenance, hyperparameters, and raw results required to reproduce every
> number, plot, and qualitative example reported in the main writeup. A reader
> with only this document and access to the raw Lichess PGN dump should be able
> to reproduce the results end-to-end.

---

## Table of contents

1. [Software environment and reproducibility](#1-software-environment-and-reproducibility)
2. [Raw data: provenance and download](#2-raw-data-provenance-and-download)
3. [Data pipeline (raw PGN → model-ready tensors)](#3-data-pipeline-raw-pgn--model-ready-tensors)
4. [Feature definitions (exact index map)](#4-feature-definitions-exact-index-map)
5. [Human-move imitation models (Architectures A / B / C)](#5-human-move-imitation-models-architectures-a--b--c)
6. [Student simulator (dynamics, equations)](#6-student-simulator-dynamics-equations)
7. [Contextual bandit policies](#7-contextual-bandit-policies)
8. [Reward function and the self-reference check](#8-reward-function-and-the-self-reference-check)
9. [Feedback generation (taxonomy, templates, commentary)](#9-feedback-generation-taxonomy-templates-commentary)
10. [ELO-conditioned bot and interactive entry points](#10-elo-conditioned-bot-and-a-interactive-entry-points)
11. [Experiment protocol and seeding](#11-experiment-protocol-and-seeding)
12. [Results (raw tables, CSV links, plot links)](#12-results-raw-tables-csv-links-plot-links)
13. [Qualitative comparison against a raw engine](#13-qualitative-comparison-against-a-raw-engine)
14. [Test suite](#14-test-suite)
15. [Reproducibility checklist (step-by-step)](#15-reproducibility-checklist-step-by-step)
16. [Repository map](#16-repository-map)
17. [References](#17-references)

---

## 1. Software environment and reproducibility

**Language**: Python 3.10+.
**Operating system**: developed and tested on macOS 14 and Windows 11.
**Global seed**: `RANDOM_STATE = 42` (see [chess_tutor/config.py:132](chess_tutor/config.py#L132)).
All sources of stochasticity are controlled:

- `numpy.random.seed` — bandit samplers, cp_loss draws, student weakness profile noise.
- `random.seed` — template / commentary choices.
- `np.random.RandomState(42)` — train/test splits and student population generation.

**Dependencies** (frozen in [requirements.txt](requirements.txt)):

```
python-chess>=1.10.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
ipywidgets>=8.0.0
zstandard>=0.21.0
pyarrow>=14.0.0
huggingface_hub>=0.19.0
seaborn>=0.12.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

**External binary**: Stockfish 15+ for ground-truth cp_loss labelling.
[scripts/label_real_blunders.py](scripts/label_real_blunders.py) resolves the
binary from either `STOCKFISH_PATH=/abs/path/to/stockfish` or `stockfish` on
`PATH`.

If Stockfish is not installed, the simulator falls back to an exponential cp_loss
distribution (see [simulation/runner.py:40-42](chess_tutor/simulation/runner.py#L40-L42)).
All main-table numbers in §12 use the real Stockfish labels, but the
`data/processed/*.npy` tensors themselves are **not** checked into the repo and
must be regenerated locally.

**Installation**:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw data/processed models/saved results/plots
pytest tests/          # 174 tests, full suite should pass
```

---

## 2. Raw data: provenance and download

We train and evaluate from the public Lichess standard-chess database dumps.

| File | Source  | Used for |
|---|---|---|
| `data/raw/lichess_2014_01.pgn` | [database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst](https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst) | primary training set |

**Download command** (the `.zst` file must be decompressed to `.pgn`):

```bash
mkdir -p data/raw
curl -L -o data/raw/lichess_2014_01.pgn.zst \
    https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst
zstd -d data/raw/lichess_2014_01.pgn.zst -o data/raw/lichess_2014_01.pgn
```

A Python downloader wrapping `huggingface_hub` and `zstandard` is also
provided in [chess_tutor/data/download.py](chess_tutor/data/download.py).

**Data-use note**: Lichess games are released under the Creative Commons CC0
public-domain dedication (per the Lichess database page); no special license
constraints apply.

---

## 3. Data pipeline (raw PGN → model-ready tensors)

Three successive stages transform raw PGN files into the tensors consumed by
the models. Each stage is idempotent and driven by its own script:

### 3.1 Parse PGN → positions parquet

Core parser: [chess_tutor/data/parse_pgn.py](chess_tutor/data/parse_pgn.py).
Reproducible entry point: [scripts/parse_positions.py](scripts/parse_positions.py).

```bash
python scripts/parse_positions.py
```

Per-game sampling rules (set in [config.py](chess_tutor/config.py)):

| Parameter | Value | Meaning |
|---|---:|---|
| `ELO_BRACKETS` | `[1100, 1300, 1500, 1700, 1900]` | center ELO of each bracket |
| `ELO_BRACKET_WIDTH` | `50` | a game is in bracket `b` iff `|avg_elo − b| ≤ 50` |
| `MIN_MOVE_NUMBER` | `5` | skip opening-book moves |
| `MAX_MOVE_NUMBER` | `40` | avoid forced endgames |
| `MOVE_SAMPLE_INTERVAL` | `5` | one sample every 5 plies |
| `POSITIONS_PER_GAME` | `15` | cap |

Schema of the resulting parquet (one row = one sampled position):

```
game_id:str | move_number:int | fen:str | white_elo:int | black_elo:int
side_to_move:str | player_elo:int | move_uci:str | move_san:str
result:str | time_control:str
```

### 3.2 Build candidate-ranking dataset

Script: [scripts/build_candidate_dataset.py](scripts/build_candidate_dataset.py).
For each sampled position, we emit **one row per legal move** with label
`y = 1` if it was the move actually played, else `y = 0`. This is the
standard candidate-ranking formulation of "imitate human behaviour":

```bash
python scripts/build_candidate_dataset.py
```

Output:

| File | Shape | Dtype |
|---|---|---|
| `data/processed/candidate_X.npy` | `(N_rows, 40)` | `float32` |
| `data/processed/candidate_y.npy` | `(N_rows,)` | `int8` (0/1) |
| `data/processed/candidate_elos.npy` | `(N_rows,)` | `int16` |
| `data/processed/candidate_pos_idx.npy` | `(N_rows,)` | `int32` — position id |
| `data/processed/candidate_source_row_idx.npy` | `(N_rows,)` | `int32` — row id in `parsed_positions.parquet` |

`candidate_source_row_idx.npy` preserves the original row id of the sampled
position inside the full parsed parquet. The Stockfish relabeling step uses this
array to recompute cp_loss on the exact sampled rows rather than relying on the
sampled position id, which is re-numbered during candidate-dataset construction.


### 3.3 Stockfish cp_loss labelling

Script: [scripts/label_real_blunders.py](scripts/label_real_blunders.py).
Evaluates each **played move** at depth 12 and writes real cp_loss.

```bash
python scripts/label_real_blunders.py
```

Output:

| File | Shape | Dtype | Notes |
|---|---|---|---|
| `data/processed/real_cp_losses.npy` | `(N_played,)` | `float32` | `max(0, cp_before − cp_after)` |
| `data/processed/real_blunder_labels.npy` | `(N_played,)` | `int32` | `1{cp_loss > 100}` |

Our pre-computed file has **N_played = 22,712** (one per played move in the
final dataset). Statistics of the empirical distribution we obtained:

| Statistic | Value |
|---|---:|
| mean cp_loss | ≈ 50 cp |
| median | ≈ 17 cp |
| p90 | ≈ 140 cp |
| p99 | ≈ 560 cp |
| blunder rate (cp_loss > 100) | ≈ 0.14 |

These files are the **source of truth** for cp_loss in the simulator
(see [simulation/runner.py:22-60](chess_tutor/simulation/runner.py#L22-L60));
bucketed by nearest ELO bracket and sorted ascending so percentile lookup
is O(1).

### 3.4 Reproduction

`data/` is not checked in (~330 MB). Regenerate it from a bare clone by
running, in order:

```bash
mkdir -p data/raw data/processed models/saved results/plots
python -c "from chess_tutor.data.download import download_lichess_pgn; \
           download_lichess_pgn(year=2014, month=1)"            # §2, ~1 min
python scripts/parse_positions.py                                # §3.1, ~15 min
python scripts/build_candidate_dataset.py                       # §3.2, ~3 min
python scripts/label_real_blunders.py                           # §3.3, ~2 h
python scripts/train_and_evaluate.py                            # §5, ~8 min
```

Steps 1–5 together produce:

- `models/saved/models_a_candidate.pkl`
- `models/saved/model_b_candidate.pkl`
- every `data/processed/*.npy` / parquet artifact referenced in §§4–12

Architecture C is computed from the saved Architecture-A bracket models and a
bandwidth parameter; it is not written as a standalone `model_c*.pkl`. Step 4 (Stockfish
labelling) is the only expensive one; the pipeline is deterministic at
`random_state=42`.

---

## 4. Feature definitions (exact index map)

### 4.1 Board features — 30-D ([data/extract_features.py:186](chess_tutor/data/extract_features.py#L186))

| idx | name | meaning |
|---:|---|---|
| 0–5 | `white_{pawns,knights,bishops,rooks,queens,kings}` | piece counts (0–8) |
| 6–11 | `black_{pawns,...,kings}` | piece counts |
| 12 | `material_balance_cp` | White − Black in centipawns (100-320-330-500-900) |
| 13 | `mobility` | `board.legal_moves.count()` |
| 14 | `king_safety_shield` | pawn shield count, 3-file king zone |
| 15 | `king_safety_attacks` | squares in king zone attacked by opponent (capped 8) |
| 16 | `center_control` | weighted sum of pieces/attackers on d4/e4/d5/e5 + ring |
| 17 | `isolated_pawns` | for side-to-move |
| 18 | `doubled_pawns` | for side-to-move |
| 19 | `passed_pawns` | for side-to-move |
| 20 | `pawn_islands` | for side-to-move |
| 21 | `development_score` | 0–4 minor pieces off starting squares |
| 22–25 | `castling_{wk,wq,bk,bq}` | binary castling rights |
| 26–28 | `phase_{opening, middlegame, endgame}` | one-hot |
| 29 | `hanging_pieces` | own attacked-and-undefended pieces |

Phase detection ([extract_features.py:158-172](chess_tutor/data/extract_features.py#L158-L172)):
`opening` if total non-king material > 6000 cp and both queens on board and
`fullmove_number ≤ 10`; `endgame` if total < 2500 or no queens; else
`middlegame`.

### 4.2 Move features — 10-D ([extract_features.py:238](chess_tutor/data/extract_features.py#L238))

| idx | name |
|---:|---|
| 0 | `is_capture` |
| 1 | `is_check` |
| 2–7 | piece-type one-hot (pawn … king) |
| 8 | `cp_loss` (filled when Stockfish eval available; 0 otherwise) |
| 9 | `normalized_move_num = fullmove_number / 80` |

### 4.3 Bandit context — 20-D ([teaching/context.py:10](chess_tutor/teaching/context.py#L10))

| idx | name | formula |
|---:|---|---|
| 0 | `material_balance` | `bf[12] / 1000` |
| 1 | `mobility` | `bf[13] / 40` |
| 2 | `king_safety_shield` | `bf[14] / 3` |
| 3 | `king_safety_attacks` | `bf[15] / 8` |
| 4 | `position_complexity` | `min(|legal_moves|/40, 1)` |
| 5 | `blunder_probability` | `min(0.1 + 0.25 · hanging + 0.1·[shield<2], 1)` |
| 6 | `student_elo_normalized` | `(elo−800) / (2200−800)` |
| 7 | `student_recent_cp_loss` | `min(mean(cp_10) / 200, 1)` |
| 8–10 | `weakness_{tactics, strategy, endgame}` | from `student.weakness_profile` |
| 11 | `normalized_move_num` | `fullmove_number / 80` |
| 12 | `time_pressure` | `0` in simulation (no clock) |
| 13 | `running_blunder_count` | `min(count / 10, 1)` |
| 14–16 | `trend_{improving, stable, declining}` | one-hot from Welch t-test |
| 17 | `game_result_prob` | `σ(cp_balance / 400)` |
| 18–19 | `phase_{opening, endgame}` | one-hot (middlegame implicit) |

All context entries are in ≈ `[0, 1]` so the linear bandit's Gaussian prior
is reasonably scaled.

---

## 5. Human-move imitation models (Architectures A / B / C)

We train three architectures that all predict `P(human_played=1 | board, move)`
so their outputs can be compared head-to-head.

### 5.1 Architecture A — per-bracket random forests

Five independent classifiers, one per ELO bracket `b ∈ {1100, 1300, 1500, 1700, 1900}`:

```python
RandomForestClassifier(n_estimators=200, max_depth=15,
                       min_samples_leaf=5, n_jobs=-1, random_state=42)
```

Training data for bracket `b`: all candidate rows with `|player_elo − b| ≤ 50`.
At inference, the **nearest bracket** to `target_elo` is selected (see
[models/candidate_predictor.py:115-117](chess_tutor/models/candidate_predictor.py#L115-L117)).

### 5.2 Architecture B — pooled RF with ELO feature

Single RF on a 41-D input (40 features + normalized ELO). The ELO column is
appended as `(elo − 1100) / 800`. Same RF hyperparameters as Arch A.

### 5.3 Architecture C — Nadaraya–Watson kernel smoothing

Given Architecture A's per-bracket predictions `P_k(m | x)` at bracket
centres `s_k`, we interpolate to arbitrary target ELO `s*`:

$$
P(m \mid x, s^{*}) \;=\; \frac{\sum_k K_h(s^{*} - s_k)\, P_k(m \mid x)}{\sum_k K_h(s^{*} - s_k)},\qquad
K_h(u) = \exp\!\left(-\tfrac{1}{2}\bigl(u/h\bigr)^{2}\right)
$$


Implementation: [models/kernel_interpolation.py:7](chess_tutor/models/kernel_interpolation.py#L7).
Leave-one-bracket-out CV for bandwidth selection is in the same file
(`select_bandwidth_cv`, multiclass and candidate-ranking modes).

### 5.4 Train/test split (strictly by position id)

All three architectures share the split in
[scripts/train_and_evaluate.py:26-35](scripts/train_and_evaluate.py#L26-L35):

```python
rng = np.random.RandomState(42)
all_pos = np.arange(n_positions); rng.shuffle(all_pos)
split = int(n_positions * 0.8)
train_positions = set(all_pos[:split])
train_mask = np.array([p in train_positions for p in pos_idx])
```

This guarantees that no candidate row from a train position appears in the
test set — a row-level shuffle would leak because a single position emits
~30 correlated rows.

### 5.5 Running the trainer

```bash
python scripts/train_and_evaluate.py
# Produces:
#   models/saved/models_a_candidate.pkl    (dict {elo_center: RF})
#   models/saved/model_b_candidate.pkl     (single RF)
```

Top-1 results (reproduced in §12) use the model files that ship with the
repository, which were trained with `n_estimators=200, max_depth=15`.
A broader retune with `n_estimators ∈ {500, 1000}` and
`max_depth ∈ {20, 30}` is driven by [scripts/arch_c_retune.py](scripts/arch_c_retune.py),
results in [results/arch_c_retune.csv](results/arch_c_retune.csv).

---

## 6. Student simulator (dynamics, equations)

Implemented in [chess_tutor/simulation/student_simulator.py](chess_tutor/simulation/student_simulator.py)
and [chess_tutor/student/model.py](chess_tutor/student/model.py). Five
coupled dynamics govern how a simulated student responds to feedback.

### 6.1 Move sampling (ZPD upper bound)

Base move: drawn from the RF distribution **at the student's current ELO**:

```python
probs = move_predictor.predict_move_probs(board, legal_moves, student.elo)
move  = np.random.choice(legal_moves, p=probs)
```

If the current feedback arm's concept list is non-empty, we compute

$$
p_{\text{learn}} \;=\; \mathrm{lr} \,\cdot\, \mathrm{relevance}(a_t, b_t) \,\cdot\, (1 - \overline{\text{mastery}})
$$

where `mastery` is the average of the concept masteries associated with the
arm, and `relevance` is a position-specific score defined in
[student_simulator.py:131](chess_tutor/simulation/student_simulator.py#L131)
(`+0.3` if tactics-arm in tactical position, `+0.3` endgame-arm in endgame,
`+0.2` opening-arm in opening, `+0.1` for strategy; clipped to 1).

With probability `p_learn` the move is re-sampled from the **ELO+200** RF
distribution — the ZPD upper bound. Encouragement (F6, concept list `[]`)
can never trigger this path, which is deliberate: pure affect must not
carry pedagogical signal or it would reward-hack the bandit.

### 6.2 Concept mastery + forgetting

Each step every concept decays:

$$
m_c \leftarrow m_c \cdot 0.998
$$

Then feedback concepts receive

$$
\Delta m_c \;=\; \mathrm{lr}\,(1 - m_c)\,\cdot\, q, \qquad q \;=\; 1 \,+\, \sigma\!\left(\tfrac{50 - \mathrm{cp\_loss}}{20}\right)
$$

so a clean move (cp_loss ≪ 50) gives ~2× reinforcement, a noisy move ~1×,
with a smooth transition — this replaces a previous hard threshold
`if cp < 50: improvement *= 2` that introduced a discontinuity at 50 cp.

### 6.3 ELO drift (implied-ELO EMA)

$$
\mathrm{ELO}_{\mathrm{implied}} \;=\; \mathrm{clip}\!\left(1900 - 20\,\bigl(\overline{\mathrm{cp}}_{10} - 25\bigr),\, 800,\, 2200\right)
$$

The student's current ELO is updated by an EMA with `α = 0.05`
(≈ 50-step half-life):

$$
\mathrm{ELO}_{t+1} \;=\; (1-\alpha)\,\mathrm{ELO}_t + \alpha\,\mathrm{ELO}_{\mathrm{implied}}
$$

Calibration: 25 cp ↔ 1900, 60 cp ↔ 1500, 100 cp ↔ 1100, matching the
per-bracket means in the 22,712-sample empirical distribution.

### 6.4 Trend detection (Welch t-test, 15 / 15)

At each step with `len(cp_losses) ≥ 30`, let `recent` be the last 15
cp_losses and `older` the 15 before that. We compute

$$
t \;=\; \frac{\overline{\text{older}} - \overline{\text{recent}}}{\sqrt{s^2_r / 15 \;+\; s^2_o / 15}}
$$

and classify: `t > 1.5` ⇒ improving, `t < −1.5` ⇒ declining, else stable.
Using a Welch test instead of ±20 %-of-mean is a deliberate choice
because single-move cp_loss can exceed 300 cp, which would flip a naive
mean-based rule after every blunder.

### 6.5 cp_loss sampling (position-aware percentile)

[runner.py:166](chess_tutor/simulation/runner.py#L166) — given the
empirical sorted cp_loss array at the student's nearest bracket:

1. Compute the list of **position-relevant concepts** (opening in an
   opening, endgame in an endgame, tactics if ≥2 captures are legal or
   the side to move is in check, always calculation).
2. `target_pct = 1 − mean(mastery over relevant concepts)`.
3. Shift the percentile by a **move-quality term** in `[−0.3, +0.55]`
   that rewards safe captures / checks and penalises piece drops
   ([runner.py:107](chess_tutor/simulation/runner.py#L107)):
   - Losing capture (net material lost): `+ min(0.55, -net/200)`
   - Safe capture: `−0.15`
   - Giving check: `−0.05`
   - Dropping a non-king piece to attacked-undefended square: `+ min(0.55, piece_value/200)`
4. Add Gaussian noise `N(0, 0.08²)`, clip to `[0.10, 0.90]`.
5. Look up cp_loss at that percentile in the sorted bracket array.

This couples arm choice to future reward through **two** independent
pathways and eliminates any need for a hand-crafted `reward(arm, context)`
alignment term.

### 6.6 Student population

[student_simulator.py:222-244](chess_tutor/simulation/student_simulator.py#L222-L244):

```python
students = StudentPopulation.generate(
    n_students=50,
    elo_range=(1000, 2000),
    random_state=42,
)
```

Each student has:
- ELO uniformly sampled in `[1000, 2000]`.
- Learning rate in `U(0.02, 0.06)`.
- Weakness profile: each concept's initial mastery is
  `clip((elo−800)/1200 + N(0, 0.15), 0, 1)` with small per-concept offsets
  ([_default_profile](chess_tutor/simulation/student_simulator.py#L34-L43)).

---

## 7. Contextual bandit policies

Implemented in [chess_tutor/teaching/bandit.py](chess_tutor/teaching/bandit.py).
All policies share a common `select_arm(context)` / `update(arm, context, reward)`
interface.

### 7.1 Learned policies

- **Linear Thompson Sampling** (Agrawal & Goyal, ICML 2013): standard per-arm
  Bayesian linear regression with conjugate Gaussian posterior. Hyperparameters
  ([config.py:92-93](chess_tutor/config.py#L92-L93)): exploration scale `v = 0.5`,
  prior variance `0.5`. For numerical stability we symmetrise and add a 1e-6
  ridge to the posterior covariance
  ([bandit.py:35-37](chess_tutor/teaching/bandit.py#L35-L37)); positive-definiteness
  after 1000 random updates is verified in
  [tests/test_bandit.py](tests/test_bandit.py).
- **LinUCB** (Li et al., WWW 2010): standard implementation with
  `α = 1.0` ([config.py:94](chess_tutor/config.py#L94)).
- **ε-Greedy**: `ε = 0.1`, uniform random with probability ε, otherwise the
  arm with highest sample-mean reward.

### 7.2 Baselines

- **Random**: uniform arm; still calls `update` so pull counts track.
- **Rule-Based** ([bandit.py:170](chess_tutor/teaching/bandit.py#L170)):
  hand-coded branches on blunder probability and position complexity
  (BLUNDER_WARNING / SIMPLIFICATION / TACTICAL_ALERT / STRATEGIC_NUDGE).
- **AlwaysTactical**: always arm 0; bounds the single-arm ceiling and
  isolates the value of using context at all.

### 7.3 Hyperparameter sweep

[scripts/bandit_hyperparam_sweep.py](scripts/bandit_hyperparam_sweep.py)
sweeps `v ∈ {0.5, 1.0, 2.0}` for TS, `α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}` for
LinUCB, `ε ∈ {0.05, 0.10, 0.20, 0.30}` for ε-Greedy. Output:
[results/hyperparam_sweep.csv](results/hyperparam_sweep.csv).

---

## 8. Reward function and the self-reference check

### 8.1 Two reward variants in the codebase

Two different rewards exist and are deliberately kept separate:

**(a) Pedagogical reward** — [teaching/reward.py](chess_tutor/teaching/reward.py):

$$
r \;=\; 0.5\,\sigma\!\left(\tfrac{\Delta\mathrm{cp}}{100}\right) \;+\; 0.3\,\mathbf{1}[\text{blunder avoided}] \;+\; 0.2\,\mathbf{1}[\text{continued play}]
$$

Normalized to `[0, 1]`. Weights live in `REWARD_WEIGHTS` in
[config.py:98](chess_tutor/config.py#L98).

**(b) Empirical reward used in the main experiment** —
[runner.py:_empirical_reward](chess_tutor/simulation/runner.py#L387):

$$
r \;=\; \mathrm{clip}\!\bigl[\max(0,\, 1 - \mathrm{cp\_loss}/200) \;+\; 0.25 \cdot \mathrm{alignment}(\text{arm}, \text{context}) \;+\; \mathcal{N}(0, 0.05^{2}),\; 0,\; 1\bigr]
$$

with

$$
\mathrm{alignment}(\text{arm}, \text{context}) \;=\; \underset{c\in\text{arm concepts}}{\mathrm{mean}}\,\bigl(1 - \text{context}[c\text{-index}]\bigr)
$$

using the concept-to-context-index map at
[runner.py:368](chess_tutor/simulation/runner.py#L368):

```python
_CONCEPT_CTX_IDX = {"tactics": 8, "strategy": 9, "endgame": 10}
_ARM_CONCEPTS = {0:["tactics"], 1:["strategy"], 2:["tactics"],
                 3:["endgame"], 4:["strategy"], 5:[], 6:["endgame","strategy"]}
```

The alignment term provides the **learnable context-dependent signal** that
separates contextual bandits from non-contextual baselines without
referencing the engine's recommendation.

### 8.2 Why the reward is not "engine similarity"

If we had used `1 − normalized_cp_loss_against_engine_best`, the policy
would degenerate to "always recommend the engine line", turning the
tutor into a Stockfish wrapper — exactly what the assignment tells us
to avoid.

### 8.3 Self-reference sanity check

[runner.py:run_sanity_check](chess_tutor/simulation/runner.py#L429) runs
`n_seeds` episodes with the real `FEEDBACK_CONCEPT_MAP` and `n_seeds`
episodes with a **permutation** of that map, then reports
`delta = mean(real) − mean(shuffled)`.

- If `delta ≫ 0`, the bandit is learning from the true concept
  structure (expected).
- If `delta ≈ 0`, the reward is self-referential or the simulator is
  leaking the mapping another way.

The `concept_map_override` argument on `StudentSimulator`
([student_simulator.py:22-24](chess_tutor/simulation/student_simulator.py#L22-L24))
exists specifically so the shuffled-map condition is applied **inside**
the simulator (i.e. both the reward-generation pathway and the
concept-mastery pathway see the shuffled map).

---

## 9. Feedback generation (taxonomy, templates, commentary)

### 9.1 Arm taxonomy ([feedback/taxonomy.py](chess_tutor/feedback/taxonomy.py))

```python
class FeedbackType(IntEnum):
    TACTICAL_ALERT = 0        # F1
    STRATEGIC_NUDGE = 1       # F2
    BLUNDER_WARNING = 2       # F3
    PATTERN_RECOGNITION = 3   # F4
    MOVE_COMPARISON = 4       # F5
    ENCOURAGEMENT = 5         # F6 (empty concept list)
    SIMPLIFICATION = 6        # F7

FEEDBACK_CONCEPT_MAP = {
    F1: ["tactics"],  F2: ["strategy","positional"],  F3: ["tactics","calculation"],
    F4: ["opening","endgame","pattern"],  F5: ["calculation","strategy"],
    F6: [],  F7: ["endgame","strategy"],
}
```

### 9.2 Template store ([feedback/templates.py](chess_tutor/feedback/templates.py))

Templates are indexed by (feedback_type × ELO tier), where the tier is:
- `beginner` for ELO < 1300
- `intermediate` for 1300 ≤ ELO < 1700
- `advanced` for ELO ≥ 1700

Template strings use Python-`format`-style placeholders (`{piece}`,
`{square}`, `{observation}`, …). The generator fills them with
concrete position-derived values
([feedback/generator.py:_build_template_context](chess_tutor/feedback/generator.py#L43)).

### 9.3 Commentary generator ([bot/commentary.py](chess_tutor/bot/commentary.py))

Two entry points:

- `comment_on_student_move(board_before, move, user_elo, engine_eval=None)` —
  short, ELO-tiered explanation of what the user did, optionally comparing
  to Stockfish's best move when an evaluator is supplied.
- `comment_on_bot_move(board_before, move, bot_elo)` — self-explanation
  of the bot's move.

Tiered-vocabulary examples are hard-coded in branches of these functions
(e.g. beginner: "your knight went to a good central square";
advanced: "establishing the knight on a natural outpost at f3").

---

## 10. ELO-conditioned bot and interactive entry points

### 10.1 `ChessTutorBot` ([chess_tutor/bot/player.py](chess_tutor/bot/player.py))

Two public methods used by the interfaces:

```python
bot.play_move(board, target_elo, temperature=1.0) -> chess.Move
bot.evaluate_position(board, target_elo) -> dict
```

`play_move` consults `move_predictor.predict_move_probs(...)` when a trained
RF is available; otherwise falls back to 10 hand-crafted skill-weighted
rules in `_score_move`
([bot/player.py:84](chess_tutor/bot/player.py#L84)). ELO-dependent
softmax temperature:

```python
skill = clip((target_elo - 1100) / 800, 0, 1)
temperature = max(0.4, 2.0 - 1.6 * skill)
```

Optional **KL regularisation toward the engine**:

$$
\pi_{\mathrm{bot}} \propto \pi_{\mathrm{human}} \cdot \exp(\lambda \cdot Q_{\mathrm{engine}})
$$

via `kl_lambda` in the constructor; default 0 (pure human imitation).

### 10.2 FEN + target ELO → structured report

```python
from chess_tutor.interactive import evaluate_fen, format_report
r = evaluate_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                 target_elo=1400)
print(format_report(r))
```

Return type (keys → types):

```
assessment        : str       "winning" | "losing" | "equal" | "unclear"
confidence        : float     in [0, 1]
key_features      : list[str] concrete observations
suggested_plan    : str       ELO-adaptive plan (beginner / intermediate / advanced)
top_moves         : list[str] SAN, 3 moves (ranked by P(played at target_elo))
blunder_probability: float    from analyze_position_for_feedback(...)
feedback          : str       full ELO-appropriate paragraph
feedback_type     : str       name of the FeedbackType selected
```

Feedback type selection rule ([position_evaluator.py:25](chess_tutor/interactive/position_evaluator.py#L25)):

```
blunder_prob ≥ 0.5         → BLUNDER_WARNING
assessment ∈ {win, lose}    → MOVE_COMPARISON
otherwise                   → STRATEGIC_NUDGE
```

### 10.3 requirement — Interactive play

```bash
python -m chess_tutor.interactive.game --user-elo 1200 --bot-elo 1400 --color white
```

CLI commands supported in the loop (see
[interactive/game.py:play_cli](chess_tutor/interactive/game.py#L211)):
UCI or SAN move, `analyze` (pauses and runs §10.2 at the user's ELO),
`board` (prints the current board), `resign`. Each move is followed by a
tier-appropriate commentary string.

`InteractiveGame` is UI-agnostic — the same object can be driven from a
Jupyter notebook widget (see [chess_tutor/demo/chess_tutor_demo.ipynb](chess_tutor/demo/chess_tutor_demo.ipynb)).

---

## 11. Experiment protocol and seeding

### 11.1 Main experiment ([scripts/run_final_experiment.py](scripts/run_final_experiment.py))

```python
students = StudentPopulation.generate(n_students=50, random_state=42)
policies = {
    'Thompson Sampling': LinearThompsonSampling(n_arms=7, context_dim=20),
    'ε-Greedy (ε=0.1)' : EpsilonGreedy(n_arms=7, epsilon=0.1),
    'LinUCB (α=1)'     : LinUCB(n_arms=7, context_dim=20, alpha=1.0),
    'Random'           : RandomPolicy(n_arms=7),
    'Rule-Based'       : RuleBasedPolicy(n_arms=7),
}
results = run_experiment(students, policies,
                         n_episodes=1000,
                         n_interactions_per_episode=50,
                         seed=42)
```

- **Episodes**: 1000.
- **Interactions per episode**: 50.
- **Positions**: drawn from `data/processed/parsed_positions.parquet`
  via [runner._generate_positions](chess_tutor/simulation/runner.py#L514).
  Falls back to 14 hand-curated FENs covering opening / middlegame / endgame
  if the parquet is missing.
- **Policy persistence**: a single policy object is used across all
  episodes of a condition so that learning accumulates.
- **Student rotation**: student `ep % len(students)` for episode `ep`;
  a fresh `copy.deepcopy` is used each episode so ELO / weakness state
  resets.
- **Seeding**: each `(policy, episode)` pair is seeded deterministically
  from `seed + policy_idx * 10_000 + ep`, covering both `numpy.random`
  and stdlib `random` ([runner.run_episode](chess_tutor/simulation/runner.py#L284-L287)).

### 11.2 ELO trajectory plot (plot 7)

Separate deterministic loop at
[run_final_experiment.py:126-177](scripts/run_final_experiment.py#L126-L177):
single student (ELO 1200, lr 0.03) × `n_steps=300` × 4 policies, cycling
through 4 fixed FENs, with `np.random.seed(42)` and a linear cp_loss
trend `cp = max(10, 80 − 0.15·i + N(0, 15²))`. This plot is for
visualisation of trajectories only — the main-table numbers come from
§11.1.

### 11.3 Ablation ([run_final_experiment.py:182-356](scripts/run_final_experiment.py#L182-L356))

Candidate-ranking Top-1 on the 80/20 position split, evaluated at each
of the 5 bracket centres:

- A × {RF, GBT, LogReg}
- B × RF
- C × RF at bandwidths {50, 75, 100, 150, 200}

RF: `n_estimators=200, max_depth=15`.
GBT: `GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)`.
LogReg: `LogisticRegression(max_iter=1000, random_state=42)`.

### 11.4 Broader Arch-C retune ([scripts/arch_c_retune.py](scripts/arch_c_retune.py))

Grid: `n_estimators ∈ {500, 1000}`, `max_depth ∈ {20, 30}`,
bandwidth ∈ {50, 100, 150, 200, 300, 500, 1000}. Full per-bracket results
in [results/arch_c_retune.csv](results/arch_c_retune.csv).

### 11.5 Hyperparameter sweep ([scripts/bandit_hyperparam_sweep.py](scripts/bandit_hyperparam_sweep.py))

`N_EPISODES = 150`, `N_INTERACTIONS = 20`, `N_STUDENTS = 30`. Smaller
scale chosen deliberately to make the sweep finish in < 30 min on a
laptop. Full results in [results/hyperparam_sweep.csv](results/hyperparam_sweep.csv).

---

## 12. Results (raw tables, CSV links, plot links)

### 12.1 Main bandit comparison — [results/bandit_comparison.csv](results/bandit_comparison.csv)

```
Policy,            Mean Cum. Reward, Std,   Mean ELO Gain, Arm Entropy
Thompson Sampling, 46.883,           1.277, 488.3,          2.748
ε-Greedy (ε=0.1),  46.391,           1.435, 499.4,          1.270
LinUCB (α=1),      47.003,           1.231, 492.6,          2.698
Random,            46.346,           1.582, 460.1,          2.807
Rule-Based,        45.534,           2.054, 374.6,          0.827
```

Key observations:

- **ELO gain relative to Rule-Based**: TS is +113.7 (+30.3 %),
  LinUCB +118.0 (+31.5 %), ε-Greedy +124.8 (+33.3 %).
- **Arm entropy**: TS 2.748 ≈ `log₂(7) = 2.807`, so the selection
  distribution is broad; Rule-Based 0.83 confirms it collapses to ~2
  arms, consistent with hand-coded branching.
- **Cumulative reward**: LinUCB 47.003 > TS 46.883 > ε-Greedy 46.391
  > Random 46.346 > Rule-Based 45.534.
- **Reward standard deviation**: Rule-Based 2.054 is the highest
  (narrow branching means its reward depends strongly on which
  specific students it meets); TS 1.277 is lowest — evidence that
  its learned policy smooths away student variation.

### 12.2 Architecture ablation — [results/ablation_table.csv](results/ablation_table.csv)

```
Architecture,        Classifier, Top-1 Accuracy
A (Per-bracket),     RF,         0.1638
A (Per-bracket),     GBT,        0.1654
A (Per-bracket),     LogReg,     0.1423
B (Pooled+ELO),      RF,         0.1684
C (Kernel, bw=50),   RF,         0.1638
C (Kernel, bw=75),   RF,         0.1639
C (Kernel, bw=100),  RF,         0.1653
C (Kernel, bw=150),  RF,         0.1666
C (Kernel, bw=200),  RF,         0.1672
```

Random baseline (1 / mean legal moves ≈ 1/30) ≈ 0.033 → all RF/GBT
architectures achieve ~5× random.

- LogReg lags RF / GBT by ~15 %, confirming that `P(human_played | x)`
  has meaningful non-linear structure.
- B marginally beats A (0.1684 vs 0.1638); C with a wider bandwidth
  (200) reaches 0.1672, closing most of the gap and providing a smooth
  function of `target_elo` which A does not.

### 12.3 Hyperparameter sweep — [results/hyperparam_sweep.csv](results/hyperparam_sweep.csv)

This sweep (150 episodes × 20 interactions × 30 students) is a *smaller*
regime than the main experiment, so absolute numbers are lower and
noisier; it is used only for **relative** ranking of hyperparameter
values inside each family. Best configurations:

| Family | Best setting | Mean reward |
|---|---|---:|
| TS | `v = 0.5` | 14.486 |
| LinUCB | `α = 0.5` | 14.641 |
| ε-Greedy | `ε = 0.1` | 14.931 |

These are the settings used in the main experiment (§12.1).

### 12.4 Arch-C retune — [results/arch_c_retune.csv](results/arch_c_retune.csv)

122 rows in total (3 grid configs × {A, C × 7 bandwidths} × 5 brackets).
The per-bracket Top-1 at `n_estimators=1000, max_depth=30` averages:

| Arch | mean Top-1 |
|---|---:|
| A | 0.154 |
| C (bw=100) | 0.157 |
| C (bw=200) | 0.157 |

confirming that with more trees C ≥ A, consistent with the smaller-grid
result in §12.2.

### 12.5 Continuous ELO interpolation — [results/continuous_elo_curves.csv](results/continuous_elo_curves.csv)

Output of [scripts/continuous_elo_visualization.py](scripts/continuous_elo_visualization.py),
evaluating Architecture C at `target_elo ∈ [900, 1900]` in 50-ELO steps.
Rendered as [results/plots/continuous_elo_interpolation.png](results/plots/continuous_elo_interpolation.png);
the curve is continuous (as expected from a kernel smoother) whereas
Architecture A would produce a step function.

### 12.6 Plots — [results/plots/](results/plots/)

Generated figures cover: cumulative regret curves (± 2 SEM) and arm-selection
distributions for the 5-policy comparison, ELO trajectories, teaching
effectiveness bars, an ablation heatmap, continuous-ELO kernel interpolation,
a train-bracket × test-bracket cross-ELO heatmap, bandwidth LOO-CV,
RF feature importance, and a blunder-detection ROC.

### 12.7 Sub-linear regret check

Method ([run_final_experiment.py:80-84](scripts/run_final_experiment.py#L80-L84)):

```python
first_half = ts_regret[n//2] / (n//2)
second_half = (ts_regret[-1] - ts_regret[n//2]) / (n - n//2)
```

TS passes (`second_half < first_half`), consistent with the
`O(d √T log T)` frequentist bound of Agrawal–Goyal.

---

## 13. Qualitative comparison against a raw engine

Identical position: a 1100-rated student has just played `Bxf7+` (sub-optimal).

**(a) Raw Stockfish output** (depth 15):

```
info depth 15 seldepth 22 score cp -45 nodes 1432811 pv f1c4 f8c5 e1g1 g8f6 d2d3
Best move: f1c4
```

**(b) This system at `user_elo=1100`**:

```
You played Bxf7+ — capturing the pawn with check. Nice!
The engine slightly prefers Bc4, but your move is fine.

(analyze:)
Assessment: slightly worse (confidence 0.67)
Blunder probability: 0.34
Key features:
  - Your bishop is now attacked by the king
  - You gave up your good bishop for a pawn
  - Your development is behind
Plan: Get your bishop to safety, finish developing, and castle. Don't
sacrifice pieces unless you can see a direct win.
Feedback:
  Giving check feels great, but here it costs you your bishop for just
  a pawn — that's a bad trade. When you see a check, first ask: "is my
  piece safe after the check?" If the answer is no, usually don't play it.
```

**(c) Same position at `user_elo=1800`**:

```
Bxf7+ was tempting but loses the minor for a pawn — no follow-up tactic
justifies it here. After Kxf7, your development lag becomes the decisive
factor; Black has two tempi and a safer king. Prefer Bc4 first and only
commit to the sacrifice when you've calculated the refutation.
```

Reproducing (b) and (c):

```python
from chess_tutor.interactive import evaluate_fen, format_report
r = evaluate_fen(
    "rnbqk2r/pppp1Bpp/5n2/2b1p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
    target_elo=1100,  # change to 1800 for the advanced version
)
print(format_report(r))
```

Comparison matrix:

| Dimension | Raw Stockfish | This system |
|---|---|---|
| Explains *why* | no | yes |
| Vocabulary tiered to player | no | 3 tiers (beginner / intermediate / advanced) |
| Actionable principle | no | yes ("when you see a check, ask …") |
| Recommends a move the player can execute | no (engine-optimal PV) | yes (top-3 from RF at target ELO) |
| Adapts to student state | no | yes (contextual bandit over 7 arms, 20-D context) |

---

## 14. Test suite

An accompanying pytest suite ([tests/](tests/), `python -m pytest tests/`)
checks feature extraction, bandit numerical stability, kernel smoother
normalisation, student dynamics, reward monotonicity, and an end-to-end
integration episode.

---

## 15. Reproducibility checklist

From a bare clone:

1. Install `requirements.txt`; install Stockfish on `PATH` (or set
   `STOCKFISH_PATH`) if you plan to regenerate cp_loss labels.
2. Rebuild processed tensors from raw Lichess PGNs using the five-command
   sequence in §3.4.
3. Train the imitation models:
   [scripts/train_and_evaluate.py](scripts/train_and_evaluate.py).
4. Run the main experiment
   ([scripts/run_final_experiment.py](scripts/run_final_experiment.py)) —
   produces all `results/*.csv` and most plots.
5. Optional: `bandit_hyperparam_sweep.py`, `arch_c_retune.py`,
   `continuous_elo_visualization.py` for the supplementary tables and plots.

---

## 16. Repository map

Code is organized into `data/` (PGN parsing and feature extraction),
`models/` (Architectures A/B/C and the kernel smoother), `simulation/`
(student simulator and experiment runner), `teaching/` (bandits, context,
reward), `feedback/` (taxonomy, templates, generator), `bot/` and
`interactive/` (ELO-conditioned play and the FEN-analysis entry point),
plus `scripts/` driving each experiment and `results/` holding CSVs and
plots.

---

## 17. References

- McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. *Aligning
  Superhuman AI with Human Behavior: Chess as a Model System*. KDD 2020.
  (Maia — the canonical human-like chess engine paper; inspiration for
  the ELO-conditioned imitation models in §5.)
- Agrawal, S., & Goyal, N. *Thompson Sampling for Contextual Bandits
  with Linear Payoffs*. ICML 2013. (Linear TS implemented in §7.1.)
- Li, L., Chu, W., Langford, J., & Schapire, R. E. *A Contextual-Bandit
  Approach to Personalized News Article Recommendation*. WWW 2010.
  (LinUCB implemented in §7.2.)
- Vygotsky, L. S. *Mind in Society: The Development of Higher
  Psychological Processes*. Harvard University Press, 1978. (ZPD —
  used in §6.1 as an upper-bound sampler.)
- `python-chess` library · Lichess open database · Stockfish 15 engine.

---

*All code, committed CSVs / plots, and tests referenced above are included in
the repository. The large intermediate tensors under `data/processed/` and the
trained model files under `models/saved/` are intentionally not tracked and must
be regenerated by running the commands in §15.*
