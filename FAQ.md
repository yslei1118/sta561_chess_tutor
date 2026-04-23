# FAQ — Anticipated Questions from a Skeptical Reader
Group members: Yushu Lei, Jiahe Li, Ziyang Qin, Yiming Cui, Zherui Zhang 


### 1. Your student is simulated. Isn't the whole project circular?

This is the most important question and deserves the most careful
answer.

The concern is that if the *same* design choices that created the
tutor also created the student, then good performance of the tutor
against the student is just a consistency check, not evidence of
teaching. We took three concrete steps against this.

First, the mistake distribution of the simulated student is **not**
invented. We labeled 22,712 real moves from Lichess games with Stockfish
at depth 12, and the simulator samples from those empirical distributions
conditioned on rating. A simulated 1100-rated student makes mistakes at
roughly the size and frequency of real 1100-rated players.

Second, we explicitly ran a **self-reference sanity check**. The tutor
learns by exploiting a hidden mapping from feedback styles to chess
concepts — "blunder warning" teaches tactics and calculation, "pattern
recognition" teaches openings and endgames, and so on. In the sanity
check we randomly permuted this mapping inside the simulator. If the
tutor were merely reading its own reward back to itself, it should
still succeed. It does not: performance drops to the level of the
non-contextual random baseline. This is reported in §8.3 of the
technical appendix.

Third, the reward the tutor optimizes is **not** "how close was the
student's next move to the engine's best move." Such a reward would
trivially reward the tutor for recommending engine moves — exactly what
the project is trying to avoid. Instead, the reward combines an
empirical measure of move quality at the student's current rating with
a contextual alignment term that does not reference the engine.

None of this makes the simulator a substitute for humans. It does
rule out the simplest circular-reasoning failure modes.

---

### 2. "The tutor raised student rating by ~350 points" is a big number. How seriously should I take it?

Take it as a **relative** comparison, not an absolute prediction of
human learning. Specifically: across 1,000 simulated sessions, the
learning-based tutors produced roughly 2× the simulated rating gain of
a reasonable rule-based baseline (best learner ~350 points vs rule-based
~150 points). That delta is the meaningful quantity. The absolute
number depends on how the simulator maps mistake size to rating (an
exponential-moving-average calibrated against empirical per-bracket
means), and a different but equally defensible calibration would shift
the absolute number without changing the ranking of policies.

---

### 3. Move-prediction accuracy of about 17 percent doesn't sound impressive. Is the model actually learning anything?

For a multi-class ranking problem with ~30 candidate moves per position
and a human decision-maker whose behavior is not fully determined by
the board, 17 percent top-1 is approximately five times the random
baseline and lines up with the results reported in the Maia paper
(McIlroy-Young et al., KDD 2020) on the same kind of task. The ceiling
is bounded by irreducible human noise: two 1500-rated players in the
same position genuinely play different moves. What matters for our
pipeline is that the **distribution** over moves at a given rating is
calibrated, because that distribution is what the student sampler
consumes. We did not optimize hard for top-1.

---

### 4. Why not just fine-tune a strong engine to play "weaker"?

This is a reasonable alternative and is essentially what some commercial
products do (e.g., Stockfish with a node limit, or Lc0 weakened by
temperature). Two reasons we did not pursue it.

First, weakening a strong engine tends to produce *random* weakness —
the engine blunders in ways strong players understand but human
beginners do not make. A 1100-rated human's mistakes cluster around
specific failure modes (hanging pieces, missing simple tactics,
miscoordinating rooks) that are characteristic of that level. Training
on real human games captures this directly; weakening an engine does
not.

Second, the project is explicitly about *teaching*, not about playing.
Even a perfectly human-like opponent would not address the question of
which feedback style to give at which moment, which is the part that is
learned by the contextual bandit.

---

### 5. The tutor has seven feedback styles. Why seven, and why those seven?

The list came from the literature on chess pedagogy and common coach
behaviors: tactical alert, strategic nudge, blunder warning, pattern
recognition, move comparison, encouragement, and simplification. The
exact count is not the point — the point is that the set spans several
different teaching intents (warn vs. explain vs. reinforce vs. soothe)
so that different situations genuinely do call for different choices.
You can see this in the arm-selection entropy: the learning policies
use about six of the seven arms regularly, while the rule-based
baseline collapses to about two, which is the failure mode we wanted
to demonstrate.

Adding an eighth style would be straightforward: extend the arm set,
retrain, and re-run the experiment.

---

### 6. Why a contextual bandit instead of full reinforcement learning?

Contextual bandits optimize the immediate reward given the current
context. Reinforcement learning would optimize the long-term return
over a whole game or session, which is conceptually a better match for
teaching.

We chose the bandit framing deliberately for three reasons. (a) The
action space (7 feedback styles) and the context space (20-D summary
of student and position) are both small, and the signal-to-noise ratio
in any single move is low, so a full Markov-decision-process model
would need vastly more interaction data than we can simulate cheaply.
(b) Bandit algorithms have clean regret guarantees that we can check
empirically (the sub-linear regret check in §12.7), whereas deep RL
does not. (c) Immediate-reward teaching is defensible: a blunder
warning should help *this* move or the next one, not three moves from
now.

The cost is that we cannot plan long-horizon lesson sequences. This
is one of the extensions listed under future work.

---

### 7. The "context" is a 20-dimensional vector. Is that rich enough to represent a chess position?

No, and it does not have to be. The context is not an encoding of the
position for playing — that would indeed need to be much richer. It is
a summary of features relevant to the *teaching decision*: how
complex is the position, how recently has the student been making
mistakes, which concept is this position testing, what is the
student's rating, and so on. These are the signals a human coach would
use to decide what kind of feedback to give. The move-selection model
has access to the full 40-dimensional position-and-move representation;
the bandit uses the compressed 20-D teaching summary. Splitting the
two is deliberate.

---

### 8. Your mistake distribution is measured at Stockfish depth 12. Is that deep enough?

Depth 12 is a pragmatic choice: at 8 parallel workers it takes roughly
an hour to label our dataset, whereas depth 20 would take a day or
more for marginal gain. For the purpose of the simulator — classifying
moves as "roughly good," "slight mistake," "blunder" — depth 12 is
well above the noise floor. For publication-grade evaluation of
individual blunders (i.e., if we were using cp_loss as a ground-truth
label rather than as a distribution to sample from), we would re-run
at a higher depth.

---

### 9. What stops the tutor from reward-hacking by always giving "encouragement"?

Encouragement is the one feedback style whose concept list is empty:
in the simulator it can never teach anything, and the student's move
quality does not improve as a result of receiving it. Our reward
function does not directly reward "the student feels good," so
encouragement has no shortcut to high reward. Empirically, the
learning policies select it at roughly its base rate, not
disproportionately.

More generally: any reward function can be hacked if it is specified
poorly, and reward design is itself a research area. We chose a
reward that (a) rewards measurably better play, (b) rewards selecting
feedback that matches the student's current weaknesses, and (c) does
not reward copying the engine. That rules out the obvious failure
modes but not every imaginable one.

---

### 10. Why use a simple random forest for move prediction instead of a neural network?

Random forests were chosen for three reasons: they train in minutes
on our dataset, they give calibrated probabilities out of the box
(important for downstream sampling), and they allow feature-importance
inspection, which we use for qualitative interpretation. A
convolutional or transformer model over the board would likely give
higher top-1 accuracy — Maia (McIlroy-Young et al.) uses a ResNet —
but at much higher training cost and with a weaker interpretability
story. Because the downstream consumer of move probabilities is a
student simulator, not a tournament engine, the accuracy ceiling of
the RF has been acceptable.

---

### 11. How well does this generalize outside the rating range you trained on?

We trained on five rating buckets centered at 1100, 1300, 1500, 1700,
and 1900. The Nadaraya–Watson smoother lets us interpolate to any
rating in the 900–1900 range, and we verified the resulting
move-probability curves are continuous. We would not expect the
system to behave well above 2000 or below 800 — we have no training
data there, and player behavior at those levels is qualitatively
different (masters think structurally; novices miss basic material
threats). Extrapolation would silently produce overconfident outputs.

---

### 12. Which parts of the system are baselines rather than contributions?

To be explicit: Linear Thompson Sampling, LinUCB, and ε-greedy are
textbook algorithms, and our implementation follows the original
papers (Agrawal & Goyal 2013; Li et al. 2010). The Random Forest and
Gradient Boosting classifiers are off-the-shelf scikit-learn. The
30-D board features are standard chess-engine features (material,
mobility, king safety, pawn structure).

The **contributions** of this project are the specific combination:
(a) a student simulator whose move distribution, mistake distribution,
concept mastery, and ELO drift are all calibrated to real Lichess
data; (b) a 20-D teaching context that wires student state and
position state into a single signal for a contextual bandit; (c) a
reward function that avoids reducing to "imitate the engine"; (d) a
self-reference sanity check that catches circular-reasoning failures;
and (e) ELO-continuous move prediction via kernel smoothing over
per-bracket classifiers.

---

### 13. How confident are you in the relative ranking of TS vs. LinUCB vs. ε-greedy?

Direction yes; within-group ranking, no. Across 1,000 episodes, TS,
LinUCB, and ε-Greedy sit within ~0.5 of each other on mean cumulative
reward (43.2–43.7) with standard deviations around 5.2–5.9, so the
three bandit policies are statistically indistinguishable from each
other. All three are clearly better than Random (42.7) and Rule-Based
(40.9), and their ELO-gain gap over Rule-Based is the more robust
signal. The practical takeaway is "several standard bandit algorithms
work roughly equally well here," not "TS is provably best."

---

### 14. What's the failure mode you worry about most?

That the gap between simulated and real students is larger than our
sanity checks can detect. Real humans have memory effects across
games, emotional responses to feedback, and goal-directed behavior
(they are trying to win, not just trying to make accurate moves)
that the simulator does not capture. It is plausible that a tutor
tuned against our simulator would be suboptimal against real players
in ways we cannot anticipate from the simulator's internal reward
curves. The only real remedy is to run a study with human players,
which is the first item under future work.

---

### 15. Is anything in the repository left over from earlier iterations that no longer matters?

A handful of plots suffixed `*_phase4_small.png` are from an earlier,
smaller-scale version of the experiment and are kept in the
repository for historical comparison. The legacy multiclass move
predictor in `chess_tutor/models/move_predictor.py` predates the
candidate-ranking formulation and is retained only as a fallback when
the candidate-ranking `.pkl` files are not present. The numbers in
the results tables come from the current pipeline as documented in
the technical appendix, §§11–12.

---

### 16. How much of this can I reproduce without a GPU?

All of it. No component of this project uses a GPU. The largest
CPU-bound step is Stockfish labeling (30–90 minutes on an 8-core
laptop), and the main experiment itself runs in 15–30 minutes. Exact
per-step runtimes and the command sequence are in §15 of the
technical appendix.
