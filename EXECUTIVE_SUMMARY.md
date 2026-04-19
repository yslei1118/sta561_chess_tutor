# Executive Summary — Chess Tutor: A Chess Teacher That Adapts to You

**STA 561D Final Project** | Duke University

---

## The Problem

Learning chess is hard, and the tools available today don't help enough. Chess engines like Stockfish are incredibly strong players — stronger than any human on Earth — but they are terrible teachers. When a beginner asks "why was my move bad?", an engine replies with a cryptic suggestion like "the best move is knight to f5 followed by a twelve-move forced sequence." A beginner cannot understand that, cannot reproduce it, and learns nothing they can apply tomorrow.

A good human chess teacher does something different. They watch the student, figure out what level they are at, and tailor their explanations. To a beginner, they say: "you left your knight in a spot where I can take it for free — try protecting it first." To an intermediate player, they say: "this move is playable, but it weakens your pawn structure for later in the game." The explanation changes based on what the student can actually understand and act on. Today's chess software doesn't do this. It gives everyone the same expert-level output regardless of their skill.

This project builds a chess tutor that adapts to the student.

## Our Approach

We built a system that does three things working together.

First, it understands how humans at different skill levels actually play. Rather than always recommending the best possible move, our system learns to predict what a typical beginner would play versus what a typical intermediate or advanced player would play. This prediction is the foundation of adaptive teaching: before the tutor can help you, it needs to understand where you are. We trained our system on one hundred thousand real chess positions from an online chess database, covering five skill levels from beginner to advanced.

Second, the system breaks down every chess position into interpretable parts: how much material each side has, how safe the king is, which pieces are active, what the pawn structure looks like. This is in contrast to deep learning "black boxes" that just output a number. Because we can point to specific features — "your king's safety is low" or "your pieces lack coordination" — the tutor can give explanations a student can actually follow.

Third, and most importantly, the system learns which kind of feedback works best for each student in each situation. We designed seven types of feedback: tactical alerts, strategic hints, blunder warnings, pattern recognition, move comparisons, encouragement, and simplification advice. Rather than picking one type and sticking with it, or using rigid rules, our system treats this as a choice it has to learn, like a new tutor figuring out over time that one student responds better to warnings and another responds better to encouragement. Over many interactions, it converges on a personalized pattern of feedback that fits each student.

## How It Teaches in Practice

Imagine a 1200-rated player just moved their queen to a square where it can be captured. Our system sees this and chooses the feedback most likely to help: probably a blunder warning, phrased in simple terms — "watch out, your queen is in danger, can you see why?" If a 1900-rated player made a subtle strategic error instead, the system would choose different feedback — maybe a move comparison pointing out a better long-term plan, phrased with the vocabulary an advanced player expects.

The student sees not just the move but the reasoning, calibrated to their level. The same position produces different feedback depending on who is looking at it. A raw chess engine cannot do this. Our system can.

## What We Built and Tested

We built a full working prototype. It includes a position evaluator you can try in our notebook: paste any chess position, pick a skill level, and see what the tutor would tell a student at that level. It includes an interactive play-against-the-bot mode where you can play a game and receive running commentary and feedback after every move. Both demos work with a dropdown or click-based interface — no chess notation required.

We tested the system in several ways. We measured how accurately it predicts human moves across skill levels, compared different feedback-selection strategies against baselines, and compared the tutor's explanations side-by-side with what a raw engine would say for the same position.

## Results

Our human-move predictor correctly identifies the move a typical player at that skill level would choose roughly one time out of six on the first guess, and one time out of three when given five guesses. This is significantly better than random (which would be about one in thirty, given how many legal moves there are in a typical position). The model genuinely distinguishes skill levels — a beginner and an advanced player looking at the same position receive different likely-move predictions.

Our adaptive feedback system was tested against four baseline strategies in a simulated teaching environment. An important finding: in earlier versions of our experiment, we saw dramatic improvements — the adaptive system appeared to outperform random selection by about twelve percent. When we looked carefully, we realized part of this advantage came from how we had designed the test environment itself. After rebuilding the test to remove that bias, the apparent gap shrank substantially: a simple exploration strategy now holds a small but real edge, while the more sophisticated Bayesian approach performs on par with random selection. We report both findings honestly. The stronger point this version makes is about how much of a seeming improvement in educational AI can actually come from the evaluation setup rather than the algorithm — a cautionary tale that the project itself exemplifies.

In side-by-side comparisons, a raw chess engine's output is identical no matter who is looking at the position. Our tutor's output changes meaningfully based on skill level — different likely moves, different suggested plans, different feedback phrasing. This is the kind of adaptation that makes the difference between a world-class player (an engine) and a world-class teacher.

## Why This Matters

Chess is a popular testbed, but the idea generalizes. The same approach — predict what a learner at a given skill level would do, break the problem into interpretable pieces, and learn which type of feedback works best for which student — applies to tutoring in programming, mathematics, language learning, and any skill where the gap between expert performance and novice understanding is the core obstacle.

Current educational technology often fails because it's built around the expert, not the learner. A system that adapts to the student, rather than forcing the student to adapt to the system, is a meaningfully different design. This project shows the adaptation is possible using statistical and probabilistic methods — no massive language models or proprietary data required.

## Limitations

We tested with simulated students rather than real learners, because a one-semester project cannot run a full user study. Our simulated students learn at realistic rates and have realistic weaknesses, but real people are more complex. A pilot with ten to twenty human players, comparing our tutor against an engine-only baseline, is the obvious next step.

The system's feedback currently uses carefully crafted templates. A language model layered on top would make the phrasing feel more natural and less repetitive. We also focused on single-move feedback rather than full-game review or spaced repetition of themes, which would be useful extensions.

## Future Work

The most exciting next step is deploying this on a real platform like Lichess and running a study with actual learners. On the research side, replacing our move predictor with a neural model while keeping the adaptive-feedback structure would combine the best of both worlds: the raw prediction power of deep learning with the pedagogical structure we designed. We would also like to extend the system from single-move feedback to multi-move teaching sequences, so the tutor can plan a lesson rather than just react to each move in isolation.

## Acknowledgments

Built using open-source tools including the python-chess library, the scikit-learn machine learning toolkit, and the publicly available Lichess chess database. Our approach draws inspiration from research on human-aligned chess AI, adaptive educational systems, and sequential decision-making under uncertainty.
