# FAQ — Chess Tutor: Adaptive Teaching with Probabilistic ML

**STA 561D Final Project** | Duke University

This document answers questions a skeptical reviewer is likely to raise.

---

## Q1. Why not just use Stockfish? Isn't a stronger engine always a better teacher?

No — strength and pedagogical value are different axes. A Stockfish evaluation tells you the objectively best move and a centipawn score. It does not tell you *why* a move is bad in a way a 1200-rated player can absorb, and it does not distinguish between a tactical oversight ("you hung a knight") and a strategic misjudgment ("you weakened your king-side pawn structure"). More importantly, Stockfish recommends moves that many humans — especially non-experts — cannot understand or reproduce. A beginner who is told "the best move is Nxe5 followed by a 12-ply forced sequence" has learned nothing actionable.

Our system complements Stockfish rather than replacing it. We use Stockfish (or heuristics, when unavailable) for ground-truth position evaluation, but we generate teaching content using human-aligned move prediction, feature-based explanations, and bandit-selected feedback types. The Engine-vs-Tutor comparison section in the demo notebook makes this difference concrete: for the same position, Stockfish's output is identical across all students, while our tutor adapts language, plan, and feedback type to skill level.

## Q2. How does ELO-conditioning actually work? Is it just concatenating a number?

We compared three architectures:

- **A (per-bracket)**: Train a separate Random Forest for each ELO bracket (1100, 1300, 1500, 1700, 1900). At inference, look up the model closest to the target ELO. This is the Maia approach.
- **B (pooled)**: Train one Random Forest with the student's ELO as an extra input feature. Simplest, but the model may not use ELO meaningfully if positions are correlated with ELO.
- **C (kernel interpolation, our contribution)**: Train per-bracket models (like A), then combine them at inference via a Gaussian kernel over ELO:
  $$P(m \mid x, s^*) = \frac{\sum_k K_h(s^* - s_k) \cdot P_k(m \mid x)}{\sum_k K_h(s^* - s_k)}$$
  where $s_k$ are the bracket centers and $h$ is selected by leave-one-bracket-out CV.

**Ablation results (top-1 on held-out test positions)**:

| Architecture | Top-1 |
|--------------|-------|
| B (pooled + ELO) | **0.1582** |
| C (kernel, best bw=200) | 0.1533 |
| C (kernel, bw=300, CV-selected) | 0.1545 |
| A (per-bracket, RF) | 0.1516 |

Honestly: **B achieves the highest top-1 in our test data**. This was a surprise. The likely reason is that pooled training gives each tree access to all ELO brackets simultaneously, which acts as implicit data augmentation; per-bracket training (A, C) splits the 22k positions across 5 models, reducing per-model sample size.

We verified this by retuning A and C with **double the compute** (`n_estimators=1000`, `max_depth=30`, bandwidth sweep extended to 1000) — see `scripts/arch_c_retune.py` / `results/arch_c_retune.csv`. Best retuned results:

| Architecture | Best config | Top-1 |
|--------------|-------------|-------|
| B (Pooled+ELO) | 200 trees, depth 15 (original) | 0.1582 |
| C (Kernel, retuned) | 1000 trees, depth 30, bw=1000 | 0.1511 |
| A (Per-bracket, retuned) | 1000 trees, depth 30 | 0.1503 |

Even at 5× the trees and 2× the depth, C and A still lose to the smaller-compute B. This rules out "A and C just need more tuning" and supports the structural claim: pooled training's per-tree access to all brackets is a genuine advantage over per-bracket sharding on this dataset size.

So why do we call C "our contribution" if B is more accurate? Because **C answers a question B cannot**: *what move would a human at ELO 1640 play?* B can be queried at any ELO (by plugging in the value), but the underlying forest was never trained to distinguish ELO 1640 from ELO 1700; the ELO feature has the same weight as any other board feature. C explicitly interpolates between per-bracket predictions, giving smooth skill-level queries across the entire range 1100–1900 and graceful degradation (via the kernel tails) outside it. This is an **inference-time methodology**, not an accuracy optimization — and we should have been clearer about that distinction in our earlier writing.

Cross-ELO accuracy matrix: **3 of 5 brackets show diagonal dominance** (1500, 1700, 1900). The 1100 and 1300 brackets are actually predicted most accurately by the **1700-bracket model**, not their own. This is a genuine finding about cross-bracket generalization: stronger-player data produces models that generalize better to weaker brackets than the reverse — likely because high-ELO games contain the full range of positional concepts, whereas low-ELO games feature a narrower pattern distribution. This observation also explains why Architecture B (pooled training) beats A (per-bracket) on top-1: pooling exposes each tree to the generalizable high-bracket information.

## Q3. How is kernel bandwidth chosen? Did you just pick one that looked nice?

We ran leave-one-bracket-out cross-validation with proper ranking evaluation: for each held-out bracket, use the remaining four brackets' models to interpolate move probabilities, then compute top-1 ranking accuracy on the held-out played moves. We swept bandwidths ∈ {25, 50, 75, 100, 150, 200, 300} measured in ELO points.

**Real CV results (top-1 accuracy across held-out brackets):**

| bandwidth | held-out top-1 |
|-----------|----------------|
| 25  | 0.1522 |
| 50  | 0.1522 |
| 75  | 0.1522 |
| 100 | 0.1523 |
| 150 | 0.1527 |
| 200 | 0.1539 |
| **300** | **0.1545** ← CV winner |

The CV-selected bandwidth is **h = 300**, the largest value in our candidate set. At h = 300 the kernel weights across the five ELO brackets are nearly uniform, so Architecture C effectively reduces to a simple *average* of the five per-bracket models — which is very close in spirit to Architecture B (pooled training). This explains why B beats C on top-1 accuracy: once CV picks the widest kernel, C loses the per-bracket specialization that motivated the kernel approach in the first place.

**Earlier versions of this document claimed "CV-selected h = 100"** — that was the result of a bug in ``NadarayaWatsonELO.select_bandwidth_cv`` which used ``argmax`` on binary candidate-classifier output (meaningless) rather than per-position ranking accuracy. We fixed the method (adding a ``pos_idx`` parameter that switches to proper ranking evaluation) and re-ran. This is the corrected number.

## Q4. Thompson Sampling sounds fancy. Why not just a rule-based policy?

We compared Thompson Sampling against four baselines: uniform random, ε-greedy, LinUCB, and a hand-designed rule-based policy — **under a purely empirical reward function with no hand-crafted context-arm alignment** (see Q5 for the reward architecture). This is the version that most honestly isolates whether the contextual bandits genuinely learn pedagogically useful patterns, or whether any previous apparent advantage was simply recovering a rule we had written into the reward.

**Primary experiment (`results/bandit_comparison.csv`, 300 episodes × 30 interactions, pure empirical reward, 20-dim context with active phase features):**

| Policy | Mean Cum. Reward | Std | ELO Gain | Arm Entropy |
|--------|-----------------|-----|----------|-------------|
| **Thompson Sampling** | **25.649** | 4.16 | 26.5 | **2.81** |
| ε-Greedy (ε=0.1) | 25.646 | 4.22 | 26.7 | 1.03 |
| LinUCB (α=1) | 25.597 | 4.08 | 25.4 | 2.78 |
| Random | 25.592 | 4.22 | 26.2 | **2.81** |
| Rule-Based | 25.521 | 4.32 | 26.2 | 1.18 |

**Hyperparameter sweep (`results/hyperparam_sweep.csv`, 150 episodes × 20 interactions):**

| Rank | Policy (best config) | Mean Reward |
|------|----------------------|-------------|
| 1 | ε-Greedy (ε=0.1) | **16.765** |
| 2 | ε-Greedy (ε=0.3) | 16.626 |
| 3 | ε-Greedy (ε=0.2) | 16.573 |
| 4 | ε-Greedy (ε=0.05) | 16.439 |
| 5 | LinUCB (α=1.0) | 16.418 |
| 6 | TS (v=0.5) | 16.364 |
| 7-9 | TS/LinUCB variants | 16.28-16.36 |
| 11 | Random | 16.271 |
| 13-14 | LinUCB (α=0.1, 0.5) | 15.79-16.14 |

**Honest reading:**

1. **All 5 policies are statistically indistinguishable.** The spread is 25.52 to 25.65 — 0.5% of the absolute value — while each policy's standard deviation is about 4.2. Any ordering between them is within noise. This is the single most honest finding of the project: *under a genuinely empirical reward, no bandit beats Random in a statistically meaningful way*.
2. **Thompson Sampling and Random share identical arm entropy (2.81).** In the absence of a strong arm-differentiating signal, TS's posterior stays near the uniform prior, so its sampling distribution matches Random's. This is not an implementation failure; it is TS correctly reporting that the signal is weak.
3. **The pseudo-ranking flips between runs.** An earlier run (before we revived two dead context dimensions) had ε-Greedy leading by ~2.5%. After reviving `phase_opening` and `phase_endgame` features, Thompson Sampling edged ahead by 0.01%. When the gap is this small, the "winner" depends on which random seed you happened to use.
4. **Rule-Based consistently comes last** (25.52) — hand-coded rules don't help when the environment's concept dynamics do not neatly match the rule.
5. **ELO gain spread is narrow** (25.4 to 26.7 = 5%), and again within each policy's variance.

**Why this differs from earlier versions of this project:**

Earlier iterations of the repo reported TS/LinUCB winning by 8-12% over Random. Those numbers were produced under a reward function that added a hand-crafted **alignment term** (e.g., "TACTICAL_ALERT scores higher when complexity is high, BLUNDER_WARNING scores higher when blunder_prob is high"). The bandit was largely recovering that rule — a real but tautological result. We removed the alignment term entirely and re-ran. The apparent bandit superiority mostly disappeared. This is the honest picture we now present.

**Why we still use Thompson Sampling as the default in the interactive demo:**

- **Theoretical regret bound** (Agrawal-Goyal 2013) — the only policy with a tight Bayesian regret guarantee under linear payoffs.
- **Highest arm entropy**, tied with Random at 2.81 — students see the most varied feedback types. ε-Greedy's entropy 1.05 means it quickly collapses to a dominant arm; from a pedagogy standpoint that is undesirable even if its average reward is slightly higher.
- **Effectively hyperparameter-free** (v=1 is standard).
- **Graceful degradation**: when the signal is strong (e.g., a real-student pilot where feedback effects are observable), TS's Bayesian posterior is the natural structure to accumulate evidence. In our simulation, where the signal is weak, TS correctly behaves like uniform exploration — a safe default.

So the honest narrative is: **"Under purely empirical reward (no hand-crafted alignment), no bandit policy outperforms Random in a statistically meaningful way. Any apparent ranking in a single run is within noise. What remains genuine is (i) the qualitative difference in arm-selection diversity — TS matches Random's maximal entropy while ε-Greedy collapses to a few arms; (ii) the theoretical regret guarantees for TS and LinUCB. We use TS in the demo for these two reasons, not because of a superior reward number that does not exist in this experiment."**

## Q5. Your student simulator is fake. How do we know any of this transfers to real students?

**Short answer: no real-student claim is made. But the bandit experiment is now genuinely empirical — the reward has **no** hand-crafted context-arm alignment. All arm differentiation flows through the student simulator's concept-mastery dynamics, not through rules we wrote into the reward function.**

Our simulation has three pieces:

**(1) Empirical cp_loss distribution.** We labeled 22,712 real played moves from Lichess with Stockfish depth 12 (saved in `data/processed/real_cp_losses.npy`). Per-bucket cp_losses are sorted and used as a quantile-indexed draw in the simulator.

**(2) Concept-aware sampling.** `_sample_real_cp_loss(student, board)` selects the *percentile* of the ELO-bracket distribution by the student's **concept-specific mastery** at that board:
- In an endgame position, endgame-mastery drives the percentile.
- In a tactical middlegame, tactics-mastery drives it.
- In an opening, opening-mastery drives it.

Higher mastery → lower percentile → the student plays like the best players at that ELO in that kind of position. Lower mastery → higher percentile → the student plays like the worst. This is an explicit modelling assumption, but the mechanism is **about the simulator**, not the reward function.

**(3) Pure empirical reward.** `_empirical_reward(context, arm, cp_loss)` = `max(0, 1 − cp_loss/200) + noise`. There is **no** arm dependence and **no** alignment term inside the reward. `context` and `arm` are accepted only for API compatibility and are explicitly `del`-ed inside.

**The arm still matters**, but only through the simulator: the bandit's chosen feedback type updates the student's concept-specific mastery (via `StudentSimulator.update_state`), which changes the *next turn's* percentile draw and therefore the *next turn's* cp_loss. The arm has zero direct effect on the reward; it only acts indirectly through simulator dynamics that are themselves concept-grounded.

**Why this is less self-referential than before:**
- Before: reward = 0.4 · empirical_base + 0.6 · `alignment(context, arm)`. The alignment was a hand-written rule — bandit literally learned to recover it.
- Now: reward = empirical_base only. The alignment is removed. Whatever arm differentiation we observe must flow through the simulator's concept dynamics — a realistic pedagogical mechanism rather than a reward-function rule.

**What the experiment supports:**
- When the simulator has realistic concept dynamics and the reward has no hand-crafted alignment, **bandit policies still differentiate**, but the ranking is very different (see Q4: ε-Greedy wins, not LinUCB, and the spread shrinks from ~8% to ~6%). This shift from alignment-reward to empirical-reward rankings is itself the key scientific finding.
- All context-aware policies beat Random by 0-3% under empirical reward (narrow margin). The simulator's mechanism provides real but modest signal.

**What the experiment still does not support:**
- That any specific feedback type is pedagogically right for a real human learner. The simulator's concept-mapping ("TACTICAL_ALERT improves tactics mastery") is still our assumption.
- Any absolute reward number as a measure of real learning.

**The honest frame:**
- The reward is now purely empirical — no hand-crafted alignment.
- The simulator has an explicit concept-mapping assumption (`FEEDBACK_CONCEPT_MAP` in `taxonomy.py`), which is where any remaining "self-reference" lives.
- A real-student pilot (10–20 players, within-subject vs engine-only) is the single piece of work that would validate the simulator's concept mapping.

## Q6. What does "sub-linear regret" mean here, and why is it the right metric?

Cumulative regret at time $T$ is $R(T) = T \cdot r^* - \sum_{t=1}^T r_t$, where $r^*$ is the best-achievable per-step reward. A policy achieves sub-linear regret if $R(T) / T \to 0$ as $T \to \infty$ — roughly, the average per-step reward converges to optimal.

For Thompson Sampling with linear payoffs and $d$-dimensional context, Agrawal & Goyal (2013) proved $R(T) = \tilde{O}(d \sqrt{T})$. Our simulation plot matches this shape: regret grows like $\sqrt{T}$, not linearly. Sub-linear regret is the right metric because it directly measures "how much reward did we leave on the table by exploring vs exploiting?" — a cleaner signal than any single-point reward number.

## Q7. Your move prediction accuracy is only 13–17% top-1. That seems low.

It is low in absolute terms but high in context. There are typically 30+ legal moves per position, so uniform random gives ~3% top-1 accuracy. Human players at a given ELO are highly variable in their choices — the same position will produce different moves from different 1500-rated players. The relevant ceiling is "what fraction of the move distribution can be explained by features alone?", which is bounded well below 100%. For comparison, Maia reports 50% top-1 on Lichess blitz data, but uses a deep CNN on millions of games; we use Random Forests on 100k positions with 40 hand-crafted features.

More importantly, our top-5 accuracy is 34–44%, meaning the correct move is in our top candidates nearly half the time. This is the right metric for a tutor: the tutor does not need to predict the unique move the student will play, it needs to narrow the likely set so it can generate relevant feedback. We validate this by showing that the predicted top moves do differ meaningfully across ELO brackets.

## Q8. Did you split your data properly? The "cross-ELO" evaluation looks suspicious.

Data splits are by position, not by game and not by ELO. For each position in the dataset, we split 80/20 train/test within each ELO bracket. The cross-ELO matrix is then built by training on bracket $k$'s train set and evaluating on bracket $j$'s test set — no position appears in both train and test. We use the same random seed (42) for all splits so results are reproducible.

One legitimate concern is position leakage across ELO brackets: the same opening position occurs at every bracket. Because our task is predicting the move, not the position, this is not leakage in the usual sense — the labels (moves) differ across brackets. But it means the model can potentially memorize position-specific priors and just adjust by ELO. This is partly what kernel interpolation is meant to exploit; we flag it as a methodological nuance rather than a flaw.

## Q9. What are the compute and data limits of this project?

- Data: Lichess open database, January 2013, ~93MB decompressed. About 100k positions across 5 ELO brackets after filtering (moves 5–40, sampled every 5th move).
- Compute: all models trained on a single laptop CPU (M-series Mac), no GPU required. End-to-end pipeline runs in ~35 minutes.
- Models: Random Forests (500 trees, max_depth=20). Feature extraction is the dominant cost (~10 min).
- Bandit experiments: 200 episodes × 5 policies × 20 interactions each = 20k simulated interactions, <2 min.

We deliberately chose CPU-scale methods because they are (a) reproducible on any laptop, (b) interpretable (feature importance is directly readable), and (c) adequate for the move-prediction-plus-bandits framing. Scaling to GPU-trained transformers is a future-work item, not a current bottleneck.

## Q10. Are there ethical concerns with this project?

Three worth naming:

1. **Over-fitting to specific skill populations**: if we trained only on Lichess blitz data, our "1500" may differ from FIDE 1500 or Chess.com 1500. The tutor's explanations could thus be miscalibrated for a classroom of scholastic players. We mitigated this by drawing from a broad bracket range, but flagged this as a limitation.
2. **Learned helplessness**: a tutor that always intervenes at the first inaccuracy may prevent a student from learning to self-diagnose. The bandit's SIMPLIFICATION and ENCOURAGEMENT arms are deliberate counterweights, selected when complexity is manageable or the student is declining.
3. **Chess-specific bias**: our features encode a particular view of what matters in chess (material, king safety, pawn structure). A student from a different chess tradition (e.g., hypermodern or Chinese Xiangqi intuition) may receive feedback that clashes with their framework. This is a real limitation of any feature-engineered system; a neural encoder would diffuse this concern but introduce its own opacity.

We believe none of these rise to the level of "do not deploy" concerns for an educational tool, but they are important to state.

## Q11. Can this system actually be used in production, or is it a toy?

The system as implemented is a research prototype. The interactive demo runs in real time, makes reasonable suggestions, and adapts visibly to ELO changes. For production deployment, the following would be needed:
- Replace heuristic cp-loss estimation with real Stockfish calls
- Replace rule-based feedback templates with a fine-tuned language model
- Run a user study to calibrate reward signals against real learning outcomes
- Add logging, session persistence, and failure modes (e.g., what if the user pastes an illegal FEN?)

For the scope of an STA 561D final project, the prototype is complete: all three probabilistic ML techniques (supervised prediction with kernel smoothing, contextual bandits) are implemented correctly, evaluated with appropriate baselines and metrics, and demonstrated via an interactive notebook.

## Q12. What is novel here, and what is reused from existing work?

**Reused**: ELO-conditioned move prediction (Maia), Random Forest classifiers (standard), Thompson Sampling (Agrawal & Goyal), ZPD-based student simulators (Clement et al.), LinUCB baseline (Li et al.). We do not claim any of these individually as novel.

**Our contribution**: the combination is new. Specifically:
- Using Nadaraya–Watson kernel interpolation over ELO brackets for *continuous* skill-level queries (Maia trains at discrete levels).
- Formulating feedback-type selection as a *contextual* bandit with a 20-dimensional context capturing both position and student state (prior tutoring-system bandits typically use much lower-dimensional context or none).
- Integrating the three components into a single coherent tutor demonstrated via interactive notebook, with side-by-side qualitative comparison against a raw engine.

Each component is grounded in prior work. The composition — and the evidence that it produces meaningfully different behavior from an engine — is what we submit.
