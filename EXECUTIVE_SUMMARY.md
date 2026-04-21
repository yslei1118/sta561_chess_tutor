# Executive Summary — Chess Tutor
Group members: Yushu Lei, Jiahe Li, Ziyang Qin, Yiming Cui, Zherui Zhang 

## The problem

Chess engines like Stockfish can beat any human, but they make poor
teachers. If a beginner asks Stockfish for advice, it will suggest a move
that requires twenty moves of calculation to understand, in a position
the beginner should never have reached in the first place. Strong engines
know *what* the best move is, but they do not know **how to teach it** to
a particular student. A good human coach does something different: they
watch how the student plays, figure out what the student is ready to
learn next, and give feedback that is neither too easy nor too hard.

The goal of this project is to build an interactive chess tutor that
behaves more like that coach than like a raw engine. Concretely, given a
student of a known skill level, the tutor should (a) play moves the
student can actually understand and respond to, (b) watch the student's
mistakes over the course of a game, and (c) choose the right **kind** of
feedback for each moment — a warning when the student is about to blunder,
encouragement when they are on a good run, a strategic nudge in a quiet
position, and so on. The system that decides which kind of feedback to
give has to learn from experience which choices actually help the student
improve, rather than following a fixed rulebook written by us.

## Our approach

We combined three ideas.

**(1) Learn to move like a human at a given skill level.** Instead of
asking an engine for its best move, we trained a model on roughly two
million moves from real online games, grouped by the rating of the player
who made them. The model learns the *distribution* of moves that, say,
1200-rated players tend to play — including their typical mistakes —
rather than the single best move. We also built a smoother on top of this
so the tutor can play at any rating, not just the five rating buckets we
trained on.

**(2) Simulate a student.** To study how feedback affects learning
without running a year-long study on real humans, we built a simulator
where a synthetic student picks moves, makes mistakes at a rate that
matches what we measured in real games, gradually internalizes the
concepts the tutor emphasizes, and slowly forgets what is not reinforced.
Their apparent skill rises or falls based on how they actually play,
measured against real-world data.

**(3) Let the tutor learn which feedback works.** The tutor has a menu of
seven feedback styles (tactical alert, strategic nudge, blunder warning,
pattern recognition, move comparison, encouragement, simplification).
Which style to use depends on the situation — the position on the board,
the student's recent form, their known weak spots, and so on. We framed
this as a **decision-learning problem**: the tutor tries a feedback
style, watches whether the student's next few moves get better, and
adjusts its preferences accordingly. We tested several well-known
learning strategies against simple baselines like "always give the same
feedback" and "follow a fixed rulebook."

## Results

Across 1,000 simulated teaching sessions of 50 moves each, the learning
tutors substantially outperformed the fixed-rule baseline. The best
learning strategy raised a simulated student's rating by roughly 490
points on average, compared to 375 points under the rule-based tutor —
an improvement of about 30 percent. The learning tutors also used a much
wider variety of feedback styles, matching the intuition that different
moments call for different coaching, while the rule-based baseline kept
falling back on the same two responses. We also verified that the system
is genuinely reading the situation and not gaming its own reward: when
we scrambled the hidden link between feedback styles and the concepts
they are supposed to teach, performance collapsed, as it should.

On the underlying move-prediction task, all of our architectures
predicted the actual human move about five times more often than random
guessing. The "smooth-across-ratings" architecture gave us the ability
to tutor at any skill level from 900 to 1900, not just at the discrete
training ratings.

Finally, the system ships with two interactive entry points: a single-
position analyzer (give it a board and a target rating, get a written
assessment and a plan) and a playable opponent that comments on each
move in vocabulary calibrated to the user's level.

## Future work

Four directions stand out.

First, the hardest limitation is that our student is simulated. A
simulator is a useful testbed, but the real question is whether this
system actually helps real learners. The natural next step is a study
with human players — even a small one — comparing sessions with the
tutor to sessions with a baseline engine or no help at all.

Second, the tutor currently learns separately for each new student. In
principle it should carry knowledge across students: what worked for
many 1400-rated players is informative for the next 1400-rated player
we meet. Adding this kind of shared learning should both improve
performance and reduce the amount of interaction needed before the
tutor "warms up."

Third, the feedback itself is assembled from templates. That is good
enough to study *which style* to pick, but the actual sentences would
feel more natural if they were generated by a language model conditioned
on the position and on the feedback style we select. The decision-
learning layer and the language layer can be trained and evaluated
separately, which is convenient.

Fourth, we do not yet use the clock. Real play involves time pressure,
and a good tutor should probably give terser, more urgent feedback when
the student is low on time and longer, more conceptual feedback when
they are not. Adding time as a context signal is a small extension with
a clear motivation.

The broader lesson is that superhuman play and good teaching are
different objectives, and designing a system around the second one —
with an explicit model of the student, an explicit menu of feedback
styles, and an explicit mechanism for learning which to use when —
produces qualitatively different behavior than asking a strong engine
to be friendly.
