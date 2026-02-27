# Chess Tutor: Literature Reference

> Key papers, what to borrow, and critical equations. Read the relevant section before implementing each phase.

---

## §1 Human Move Prediction & Skill Modeling

### Maia-1 (McIlroy-Young et al., KDD 2020) ⭐⭐⭐⭐⭐
- **Link:** https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf
- **Code:** https://github.com/CSSLab/maia-chess
- **Borrow:** Data methodology (Lichess by ELO bracket, temporal train/test split), evaluation metric (move-matching accuracy), baseline numbers (~51% at matched ELO)
- **Key insight:** Different ELO levels produce systematically different moves. A model trained at ELO 1200 predicts 1200-level moves better than a model trained at 1500.
- **Your angle:** They use AlphaZero architecture. You use handcrafted features + classical ML, making the same core observation but with interpretable models.

### Maia-2 (Tang et al., NeurIPS 2024) ⭐⭐⭐⭐⭐
- **Link:** https://arxiv.org/abs/2409.20553
- **Code:** https://github.com/CSSLab/maia2
- **Borrow:** Validates unified model > separate models; can use as baseline if time permits
- **Key insight:** Single model with ELO conditioning outperforms 9 separate Maia-1 models. Your Model B and C test this same hypothesis with classical ML.

### BP-Chess (Skidanov et al., arXiv 2504.05425, 2025) ⭐⭐⭐⭐
- **Borrow:** Feature engineering approach — validates that handcrafted strategy features + LinearSVC/Ridge can outperform Maia by ~25%. Direct precedent for your approach.
- **Key features they use:** Material, piece activity, king safety, pawn structure, space advantage, bishop pair, threats — very similar to your feature set.
- **Implementation note:** They use only 5,000 games per ELO bracket. Your 100K games per bracket should perform even better.

### ALLIE (Zhang et al., ICLR 2025) ⭐⭐⭐
- **Link:** https://blog.ml.cmu.edu/2025/04/21/allie-a-human-aligned-chess-bot/
- **Borrow:** Cite as latest SOTA for context. Their adaptive search idea inspires the KL-regularized bot.

### N-gram Chess (Zhong et al., arXiv 2512.01880, 2025) ⭐⭐⭐
- **Borrow:** Alternative lightweight approach — skill-group-specific n-grams achieve 39.1% accuracy improvement. Shows classical methods are viable.

### Anderson et al. (KDD 2016) ⭐⭐⭐
- **Key finding:** Position difficulty is a stronger predictor of errors than player skill. Supports your PositionComplexity model.

### Regan et al. (AAAI 2011) ⭐⭐
- **Borrow:** Intrinsic Performance Ratings — inferring skill from move quality. Useful for student tracking.

---

## §2 Contextual Bandits & Adaptive Teaching

### Thompson Sampling Tutorial (Russo et al., 2018) ⭐⭐⭐⭐⭐
- **Link:** https://arxiv.org/abs/1707.02038
- **Must read:** Chapters 1-4 (basics, Bernoulli, linear, contextual)
- **Key algorithm (Ch. 4 — Linear Thompson Sampling):**

```
Initialize: For each arm a = 1,...,K:
    B_a = I_d (d×d identity matrix)
    f_a = 0_d (d-dim zero vector)
    μ̂_a = 0_d

For t = 1, 2, ..., T:
    Observe context x_t ∈ R^d
    For each arm a:
        Sample θ̃_a ~ N(μ̂_a, v² · B_a^{-1})
        score_a = θ̃_a · x_t
    Play arm A_t = argmax_a score_a
    Observe reward r_t
    Update:
        B_{A_t} += x_t · x_t^T
        f_{A_t} += r_t · x_t
        μ̂_{A_t} = B_{A_t}^{-1} · f_{A_t}
```

- **Exploration parameter v:** Controls posterior sampling width. v=1 is default. Larger v → more exploration.

### Agrawal & Goyal (ICML 2013) ⭐⭐⭐⭐
- **Link:** https://arxiv.org/abs/1209.3352
- **Borrow:** Theoretical regret bound Õ(d^{3/2} √T) for your technical appendix
- **Implementation note:** Their v = R√(9d·ln(T/δ)) for theoretical guarantees, but v=1 works fine in practice.

### Lan & Baraniuk (EDM 2016) ⭐⭐⭐⭐
- **Link:** https://people.umass.edu/~andrewlan/papers/16edm-bandits.pdf
- **Borrow:** Framework for contextual bandits with student knowledge profiles as context. Directly maps to your setup: student ELO + weakness profile = context, feedback type = arm.

### De Kerpel et al. (arXiv 2602.04347, 2025) ⭐⭐⭐⭐
- **Borrow:** Contextual Thompson Sampling for personalized exercise sequences. Measures learning progress as skill level change — exactly your reward signal.
- **Key innovation they use:** Reward = change in estimated skill level after interaction. Apply similarly: reward = change in move quality after feedback.

### Clement et al. (JEDM 2015) ⭐⭐⭐⭐
- **Link:** https://jedm.educationaldatamining.org/index.php/JEDM/article/view/JEDM111
- **Borrow:** ZPDES algorithm — Zone of Proximal Development + Empirical Success. Use learning progress as reward signal. Their key insight: feedback helps most when student is NEAR but NOT AT mastery (the ZPD principle). Implement this in your student simulator.

### Li et al. (WWW 2010) — LinUCB ⭐⭐⭐
- **Borrow:** LinUCB algorithm as a baseline:

```
For each arm a:
    A_a = I_d
    b_a = 0_d
    
At each round:
    θ̂_a = A_a^{-1} b_a
    UCB_a = θ̂_a^T x + α √(x^T A_a^{-1} x)
    Play argmax_a UCB_a
    Update:
        A_a += x x^T
        b_a += r x
```

### Gerstner et al. (arXiv 2025) — CK-12 Foundation ⭐⭐⭐
- **Borrow:** Large-scale validation (1M students) that contextual policies outperform non-contextual ones when features are informative. Cite in FAQ to argue contextual bandits are worth the complexity.

### Rafferty et al. (AIED 2011, Cognitive Science 2016) ⭐⭐
- **Borrow:** Teaching as POMDP framework. Student knowledge is partially observable → bandit is a practical approximation to the full POMDP. Good for technical appendix discussion.

### Mandel et al. (AAMAS 2014) ⭐⭐⭐
- **Borrow:** Offline policy evaluation for educational games. Use Inverse Propensity Scoring (IPS) or Doubly Robust (DR) estimator to compare policies without running new experiments:

```
IPS estimator: V̂(π) = (1/n) Σ [π(a_t|x_t) / μ(a_t|x_t)] · r_t

where π is the target policy, μ is the logging policy, and r_t is the observed reward.
```

---

## §3 Blunder Detection & Position Analysis

### McGrath et al. (PNAS 2022) ⭐⭐⭐
- **Borrow:** Concept extraction from AlphaZero — probe representations with linear classifiers to detect material, mobility, king safety, etc. Validates that these concepts are learnable from game data.
- **Your application:** Your handcrafted features ARE these concepts. Use this paper to justify why material, mobility, king safety are the right features.

### Sadikov et al. (CG 2006) — Automated Chess Commentary ⭐⭐⭐
- **Borrow:** Template-based commentary generation. Their approach: analyze position → identify key features → fill templates. Direct precedent for your feedback/commentary system.

### GlickFormer (arXiv 2410.11078, 2024) ⭐⭐
- **Borrow:** Puzzle difficulty estimation using Glicko-2 ratings. Use for your puzzle recommendation system (active learning module).

---

## §4 Student Modeling & Knowledge Tracing

### Pelánek (Computers & Education 2016) ⭐⭐⭐⭐
- **Borrow:** Elo rating system adapted for education. Treat student answering puzzle correctly as "beating" the puzzle:

```
Expected score: E_s = 1 / (1 + 10^((R_item - R_student) / 400))
Update: R_student += K · (S - E_s)
where S = 1 if correct, 0 if incorrect, K = 32 (default)
```

- **Your application:** Track student ELO through puzzle performance and move quality.

### Abdi et al. (arXiv 2019) — M-Elo ⭐⭐⭐
- **Borrow:** Multi-concept Elo tracking. Each chess concept (tactics, strategy, endgame) gets its own Elo rating. Use for your weakness_profile.

### BKT — Corbett & Anderson (1995) ⭐⭐⭐
- **Borrow:** Bayesian Knowledge Tracing as alternative to Elo for student modeling:

```
4 parameters per skill:
    p(L0) = prior probability of knowing skill
    p(T)  = probability of learning skill after opportunity
    p(G)  = probability of guessing correctly without knowing
    p(S)  = probability of slipping (incorrect despite knowing)

Update after observing correct/incorrect:
    p(L_t | correct) = p(L_t) · (1 - p(S)) / [p(L_t)(1-p(S)) + (1-p(L_t))·p(G)]
    p(L_t | incorrect) = p(L_t) · p(S) / [p(L_t)·p(S) + (1-p(L_t))·(1-p(G))]
    p(L_{t+1}) = p(L_t | obs) + (1 - p(L_t | obs)) · p(T)
```

- **Implementation note:** BKT is simpler than deep KT but sufficient for this project. Per-concept BKT = your weakness_profile tracker.

---

## §5 Kernel Methods (Theoretical Foundation)

### Nadaraya-Watson Estimator (1964) ⭐⭐⭐⭐
- **Core equation:**

```
f̂(x) = Σ_i K_h(x - x_i) · y_i / Σ_i K_h(x - x_i)

where K_h(u) = (1/h) K(u/h) and K is Gaussian kernel:
K(u) = (1/√(2π)) exp(-u²/2)

For your application:
P(m | features, ELO*) = Σ_k K_h(ELO* - ELO_k) · P_k(m | features) / Σ_k K_h(ELO* - ELO_k)
```

- **Bandwidth selection:** Leave-one-out cross-validation is standard. For leave-one-bracket-out:

```
CV(h) = Σ_b [accuracy of interpolating bracket b from remaining brackets, using bandwidth h]
h* = argmax_h CV(h)
```

### Kossen et al. (2022) ⭐⭐
- **Borrow:** NW prediction head on neural features gives excellent calibration. Supports your argument that NW interpolation is well-calibrated for skill-level queries.

### Sarkar & Cooper (AIIDE 2017) ⭐⭐
- **Borrow:** Gaussian Process regression for predicting player ratings from game features. Direct precedent for using kernel methods in game-based skill estimation.

---

## §6 Interactive Systems & Bot Design

### KL-Regularized Search (Jacob et al., ICML 2022) ⭐⭐⭐
- **Link:** https://arxiv.org/abs/2112.07544
- **Key equation for your bot:**

```
π_bot(m | position) ∝ π_human(m | position, ELO) · exp(λ · Q_engine(position, m))

where:
    π_human = your Model C predictions at target ELO
    Q_engine = Stockfish evaluation of the resulting position
    λ = regularization parameter:
        λ = 0 → pure human model (plays like a human at target ELO)
        λ → ∞ → pure engine (plays optimally)
```

- **Implementation:** Compute log-probabilities from both sources, add with weight λ, softmax to get final distribution, sample.

### Kochmar et al. (AIED 2020) ⭐⭐⭐
- **Borrow:** Personalized feedback type selection significantly improves learning outcomes. Their Korbit ITS showed that matching feedback type to student context is more effective than fixed feedback. Direct motivation for your contextual bandit approach.

---

## Quick Reference: Paper → Project Phase Mapping

| Paper | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|-------|---------|---------|---------|---------|---------|
| Maia-1 (KDD 2020) | Data methodology | Baseline | — | — | — |
| Maia-2 (NeurIPS 2024) | — | Validates unified model | — | — | — |
| BP-Chess (2025) | Feature design | Validates classical ML | — | — | — |
| Nadaraya-Watson (1964) | — | **Core: kernel interpolation** | — | — | — |
| Agrawal & Goyal (2013) | — | — | — | **Core: TS algorithm** | — |
| Russo et al. Tutorial (2018) | — | — | — | **Core: implementation guide** | — |
| Lan & Baraniuk (EDM 2016) | — | — | — | Context design | — |
| De Kerpel et al. (2025) | — | — | — | Reward design | — |
| Clement et al. (JEDM 2015) | — | — | — | ZPD concept | — |
| Anderson et al. (KDD 2016) | — | — | Difficulty estimation | — | — |
| McGrath et al. (PNAS 2022) | — | Feature justification | Concept extraction | — | — |
| Sadikov et al. (CG 2006) | — | — | Commentary templates | — | Commentary |
| KL-Regularized (ICML 2022) | — | — | — | — | **Bot design** |
| Pelánek (2016) | — | — | — | Student Elo tracking | — |
| Mandel et al. (2014) | — | — | — | Offline policy eval | — |

---

## STA 561 Topic → Paper Mapping

| STA 561 Topic | Key Papers | Implementation |
|---------------|-----------|----------------|
| T1: Regression | ESL Ch. 3, Anderson (KDD 2016) | PositionComplexity (Ridge regression) |
| T2: Regularization | ESL Ch. 3-4, KL-Search (ICML 2022) | Kernel bandwidth selection, KL-regularized bot |
| T3: Kernel Methods | Nadaraya-Watson (1964), Kossen (2022) | **NadarayaWatsonELO interpolation** |
| T4: Classification | ESL Ch. 9-12, BP-Chess (2025) | Move prediction (RF, SVM, LogReg) |
| T5: Large Margin Classifiers | ESL Ch. 12 | SVM for move prediction |
| T7: Random Forests | ESL Ch. 15, BP-Chess (2025) | RF for move prediction, blunder detection |
| T9: Active Learning | Clement (JEDM 2015) | Puzzle selection via ZPD |
| T14: Contextual Bandits | Agrawal & Goyal (2013), Russo Tutorial (2018) | **LinearThompsonSampling** |
| T15: RL/MDP | Sutton & Barto, Rafferty (2011) | Bot play, teaching as POMDP |
