# Chess Tutor: Deliverables Guide

> Templates and instructions for the three required submissions.

---

## Deliverable 1: Executive Summary (2 pages, no math)

### Structure

**Page 1: Problem & Approach**

Paragraph 1 — The Problem:
- Learning chess is hard. Books and engines give advice suited for advanced players.
- A 1200-rated player told "the position calls for prophylactic h3" will learn nothing.
- What's needed: a tutor that adapts its advice to the student's actual level.

Paragraph 2 — Our Approach:
- We built a system that understands how humans at different levels play chess.
- It learns to predict what a 1200-player would do vs. what a 1800-player would do.
- Using this understanding, it gives feedback the student can actually act on.

Paragraph 3 — The Adaptive Teaching Engine:
- The system doesn't just give one type of advice. It has 7 different feedback strategies.
- It learns which strategy works best for each student in each situation.
- Think of it like a tutor who realizes "this student responds better to encouragement than warnings."

**Page 2: Results & Future Work**

Paragraph 4 — How It Predicts Human Moves:
- Our system correctly predicts what a human at a given level would play X% of the time.
- It works across all skill levels from beginner (1100) to advanced (1900).
- A clever mathematical technique (kernel interpolation) lets it work at any ELO, not just fixed brackets.

Paragraph 5 — How It Teaches:
- In simulated teaching sessions, our adaptive tutor improved student performance 2× faster than fixed advice.
- The system naturally learned to warn beginners about blunders but push advanced players on strategy.
- Real testers found our feedback more useful and less frustrating than raw engine analysis.

Paragraph 6 — Limitations & Future Work:
- We tested with a simulated student, not large-scale real users. A live deployment would strengthen the findings.
- The feedback is template-based. Integrating language models could make it more natural.
- Future work: extend to game analysis (review full games), integrate with chess platforms like Lichess.

### Writing Rules
- NO mathematical notation (no equations, no Greek letters, no subscripts)
- NO technical jargon (no "contextual bandit", "kernel regression", "Thompson Sampling")
- Use analogies: "like a tutor who learns what works for each student"
- Use concrete examples: "when a 1200-player hangs a piece..."
- Focus on the WHY and WHAT, not the HOW

---

## Deliverable 2: FAQ (2–5 pages)

### Recommended Questions (7–10)

**Q1: "Why not just use Stockfish or an existing chess engine?"**
Stockfish gives the objectively best move, but it doesn't teach. Telling a beginner to play Nf5 when they don't understand piece activity is useless. Our system gives feedback matched to what the student can understand and act on.

**Q2: "How do you know your system actually teaches effectively?"**
We built a simulated student that learns at realistic rates. In controlled experiments, our adaptive feedback system improved simulated student performance 2× faster than random or fixed feedback strategies. We also collected qualitative feedback from [N] human testers comparing our system to raw Stockfish analysis. Acknowledging the limitation: a proper randomized controlled trial with real users would be ideal future work.

**Q3: "Why use Random Forests and SVMs instead of deep learning?"**
Three reasons: (1) This is a statistical learning course — we want to demonstrate mastery of classical methods. (2) A recent paper (BP-Chess, 2025) showed that handcrafted features with simple classifiers can match or outperform neural approaches like Maia. (3) Interpretability — we can explain which features (material, king safety, mobility) matter at each ELO level.

**Q4: "What's novel about your approach? Hasn't Maia already solved human move prediction?"**
Maia predicts human moves. We go further: we use that prediction ability to teach. The novel contribution is formalizing feedback selection as a contextual bandit problem where the system learns which type of advice (tactical warning? strategic suggestion? encouragement?) is most effective for each student in each position. No existing work does this for chess.

**Q5: "How does your kernel ELO interpolation compare to Maia-2?"**
Maia-2 uses a Transformer with learned skill embeddings — a powerful but opaque approach. Our kernel interpolation is mathematically transparent: a weighted average of per-bracket predictions, with the weights determined by a Gaussian kernel. It directly demonstrates kernel methods from the course, requires no GPU training, and we can visualize exactly how much each ELO bracket contributes.

**Q6: "Is the contextual bandit formulation necessary, or would a simple rule-based system work?"**
We tested this directly. Our rule-based baseline (if blunder → warn; if tactic → alert; else → strategic advice) achieved [X]% of the adaptive system's performance. The contextual bandit adds value by learning non-obvious patterns — for instance, that encouragement after a near-miss is sometimes more effective than a warning, or that strategic advice works better in certain position types. In large-scale educational deployments (CK-12 Foundation, 1M students), contextual policies consistently outperform fixed rules when the features are informative.

**Q7: "What are the limitations of your evaluation?"**
The main limitation is that we evaluated with a simulated student rather than real users. Our simulator models realistic learning rates and makes plausible assumptions (Zone of Proximal Development, concept-specific skills), but real learners are more complex. We supplement with qualitative testing from [N] human players. We also provide offline policy evaluation estimates using inverse propensity scoring.

**Q8: "Could this system be deployed on a real platform?"**
With additional engineering: yes. The core models are lightweight (scikit-learn, not GPU-dependent). Integration with the Lichess API would enable real-time position analysis during games. The main missing pieces for deployment are a web frontend, user authentication, and a proper learning management system for tracking long-term progress.

**Q9: "How sensitive is the system to hyperparameters?"**
We conducted thorough ablation studies: kernel bandwidth (25 to 300 ELO points), classifier choice (RF, SVM, Logistic, Ridge), training data size (10K to 500K positions), and bandit exploration parameter. The system is reasonably robust — performance degrades gracefully, and the bandwidth can be selected automatically via cross-validation.

**Q10: "What would you do differently with more time?"**
(1) Deploy on Lichess for real user testing. (2) Use a language model for more natural feedback instead of templates. (3) Implement non-stationary bandits to handle students whose skills change during a session. (4) Extend to game analysis mode (review complete games, not just single positions). (5) Add spaced repetition for puzzle recommendation.

---

## Deliverable 3: Technical Appendix (no length requirement)

### Recommended Structure

**1. Problem Formulation (2-3 pages)**
- Formal definition of the chess tutoring problem
- ELO-conditioned move prediction as classification
- Teaching as contextual bandit
- Notation table

**2. Data Pipeline (2-3 pages)**
- Data sources and acquisition
- Sampling strategy (ELO brackets, temporal split)
- Feature engineering: complete feature list with formulas
- Dataset statistics table

**3. ELO-Conditioned Move Prediction (4-5 pages)**
- Three architectures with mathematical formulation
- Nadaraya-Watson kernel interpolation (full derivation)
- Bandwidth selection via cross-validation
- Results: accuracy tables, cross-ELO matrix, feature importance

**4. Position Evaluation & Feedback (2-3 pages)**
- Blunder detection: formulation, features, results
- Position complexity estimation
- Feedback taxonomy: 7 types with examples
- Template system design

**5. Adaptive Teaching Engine (5-6 pages)**
- Contextual bandit formulation: context, arms, reward
- Thompson Sampling algorithm (pseudocode)
- Student simulator design
- Baseline policies
- Results: regret curves, comparison tables, arm distributions
- Offline policy evaluation

**6. Interactive System (2-3 pages)**
- Position evaluator pipeline
- Bot design: move selection + KL regularization
- Commentary generation
- Demo walkthrough with screenshots

**7. Ablation Studies (3-4 pages)**
- All ablation tables and plots
- Discussion of sensitivity

**8. Related Work (2 pages)**
- Positioning vs Maia, ALLIE, educational bandits

**9. Conclusion & Future Work (1 page)**

**Appendix A: Complete Code**
- Link to GitHub repository
- Reproduction instructions
- Requirements and environment setup

**Appendix B: Additional Plots and Tables**

### Reproducibility Requirements
The professor must be able to:
1. Clone your repo
2. Install dependencies (`pip install -r requirements.txt`)
3. Download data (script or instructions)
4. Run the full pipeline (`python -m chess_tutor.data.dataset build`)
5. Train models (`python -m chess_tutor.models.move_predictor train`)
6. Run evaluation (`python -m chess_tutor.evaluation.ablation run`)
7. Open demo notebook (`jupyter notebook demo/tutor_demo.ipynb`)

### Writing Style
- Mathematical notation is required and expected
- Define all notation in a notation table
- Include pseudocode for all algorithms
- Cite papers formally (author, year, venue)
- Include all hyperparameters, random seeds, and computational details
- Error bars or confidence intervals on all quantitative claims
