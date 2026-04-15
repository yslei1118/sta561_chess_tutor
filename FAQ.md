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

Architecture C gives continuous skill-level queries (e.g., ELO 1640) without retraining and provides graceful degradation outside the training range. Our cross-ELO accuracy matrix shows that C matches or beats A in 4 of 5 brackets, with the kernel acting as implicit regularization.

## Q3. How is kernel bandwidth chosen? Did you just pick one that looked nice?

We ran leave-one-bracket-out cross-validation: for each bracket $k$, train on the other four and evaluate prediction accuracy at $k$ using only the kernel-weighted combination of the four trained models. We swept bandwidths ∈ {25, 50, 75, 100, 150, 200, 300} (measured in ELO points) and picked the one that maximized mean top-5 accuracy across held-out brackets. The chosen bandwidth is 100.

Smaller bandwidths (25–50) degenerate to nearest-bracket lookup, losing the smoothing benefit. Larger bandwidths (200+) over-smooth and make all brackets predict the same thing, losing skill differentiation. The CV plot in the notebook shows a clear inverted-U curve with a plateau around 100.

## Q4. Thompson Sampling sounds fancy. Why not just a rule-based policy?

We compared Thompson Sampling against five baselines: uniform random, ε-greedy (ε=0.1), LinUCB (α=1), a hand-designed rule-based policy ("if blunder probability high → BLUNDER_WARNING; if complexity high → SIMPLIFICATION; else → STRATEGIC_NUDGE"), and always-tactical. Results (with 95% CIs over 50 episodes):

- Thompson Sampling: +12.3% cumulative reward vs random
- Sub-linear regret curve (converges faster than ε-greedy)
- LinUCB is close but slightly behind
- The rule-based policy does well in early rounds but plateaus because it cannot learn from feedback

The rule-based policy encodes someone's prior beliefs about chess pedagogy. If those priors are wrong for a particular student population (and in our simulator, they partially are), no amount of experience will change the rules. Thompson Sampling starts with a neutral prior and learns the context-arm associations from data. This adaptivity is the whole point of using a bandit rather than rules.

## Q5. Your student simulator is fake. How do we know any of this transfers to real students?

Honest answer: we don't, yet. The simulator is a Zone of Proximal Development (ZPD) model where students have a latent weakness profile per concept (tactics, strategy, endgame, opening, calculation). After each interaction:
- Learning rate = base_lr × feedback_relevance × (1 − current_mastery)
- Mastery updates stochastically; ELO drifts accordingly

Key claim: even if absolute reward numbers don't transfer, the *ordering* of policies should. Thompson Sampling beats random in essentially any reasonable reward model where context-arm alignment matters — this is a property of the algorithm, not the simulator. The simulator lets us validate that our implementation works and produces the theoretically expected sub-linear regret.

The project limitations section explicitly flags that we have not run a user pilot. A small pilot (10–20 players, within-subject comparison of tutor feedback vs Stockfish-only) is the natural next step.

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
