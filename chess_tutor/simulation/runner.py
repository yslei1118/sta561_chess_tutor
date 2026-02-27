"""Episode runner and experiment harness for teaching policy evaluation."""

import copy
import chess
import numpy as np
from typing import Protocol

from ..feedback.taxonomy import FeedbackType, FEEDBACK_CONCEPT_MAP
from ..teaching.reward import compute_reward


class Policy(Protocol):
    """Interface that all bandit policies must implement."""
    def select_arm(self, context: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...


def run_episode(
    student,
    policy,
    feedback_generator,
    context_builder_fn,
    reward_fn,
    positions: list[chess.Board],
    n_interactions: int = 20,
) -> dict:
    """Run a single teaching episode.

    Returns:
        dict with rewards, cumulative_reward, arms_selected,
        student_elo_trajectory, cp_loss_trajectory
    """
    rewards = []
    arms_selected = []
    elo_trajectory = [student.elo]
    cp_loss_trajectory = []

    for t in range(min(n_interactions, len(positions))):
        board = positions[t % len(positions)].copy()

        if board.is_game_over():
            continue

        # Build context
        from ..data.extract_features import extract_board_features
        board_features = extract_board_features(board)

        blunder_prob = 0.3  # Default estimate
        complexity = min(board.legal_moves.count() / 40.0, 1.0)

        context = context_builder_fn(
            board=board,
            student_state=student.state,
            blunder_prob=blunder_prob,
            complexity=complexity,
            board_features=board_features,
        )

        # Select feedback type
        arm = policy.select_arm(context)
        arms_selected.append(arm)

        # Generate feedback
        feedback_type = FeedbackType(arm)
        feedback_text = feedback_generator.generate(
            board, student.elo, feedback_type
        )

        # Student responds with feedback
        actual_move = student.respond_to_position(
            board, feedback_type=arm, feedback_text=feedback_text
        )
        cp_loss_after = _estimate_cp_loss(board, actual_move)
        cp_loss_trajectory.append(cp_loss_after)

        # Compute reward with context-dependent signal
        # Key: reward depends on how well the feedback type matches the position
        reward = _context_dependent_reward(context, arm, cp_loss_after)
        rewards.append(reward)

        # Update policy
        policy.update(arm, context, reward)

        # Update student state
        concepts = FEEDBACK_CONCEPT_MAP.get(feedback_type, [])
        student.update_state(arm, cp_loss_after, concepts)
        elo_trajectory.append(student.elo)

    return {
        "rewards": rewards,
        "cumulative_reward": sum(rewards),
        "arms_selected": arms_selected,
        "student_elo_trajectory": elo_trajectory,
        "cp_loss_trajectory": cp_loss_trajectory,
    }


def _context_dependent_reward(context: np.ndarray, arm: int, cp_loss: float) -> float:
    """Compute reward that depends on context-arm alignment.

    This creates a ground truth that the bandit can learn:
    - TACTICAL_ALERT (0) is best when complexity is high (context[4])
    - STRATEGIC_NUDGE (1) is best for intermediate positions
    - BLUNDER_WARNING (2) is best when blunder_prob is high (context[5])
    - PATTERN_RECOGNITION (3) works well in openings/endgames
    - MOVE_COMPARISON (4) works for intermediate students
    - ENCOURAGEMENT (5) works when student is declining
    - SIMPLIFICATION (6) works in complex positions with low student skill
    """
    base_reward = max(0.0, 1.0 - cp_loss / 200.0)  # [0, 1]

    # Context-arm alignment bonus
    alignment = 0.0
    complexity = context[4]
    blunder_prob = context[5]
    student_elo_norm = context[6]
    trend_declining = context[16]

    if arm == 0:  # TACTICAL_ALERT
        alignment = 0.5 * complexity + 0.2 * (1 - student_elo_norm)
    elif arm == 1:  # STRATEGIC_NUDGE
        alignment = 0.3 * (1 - complexity) + 0.2 * student_elo_norm
    elif arm == 2:  # BLUNDER_WARNING
        alignment = 0.6 * blunder_prob + 0.1 * (1 - student_elo_norm)
    elif arm == 3:  # PATTERN_RECOGNITION
        alignment = 0.3 * student_elo_norm + 0.2 * (1 - complexity)
    elif arm == 4:  # MOVE_COMPARISON
        alignment = 0.3 * student_elo_norm + 0.2 * complexity
    elif arm == 5:  # ENCOURAGEMENT
        alignment = 0.4 * trend_declining + 0.2 * (1 - blunder_prob)
    elif arm == 6:  # SIMPLIFICATION
        alignment = 0.4 * complexity + 0.3 * (1 - student_elo_norm)

    reward = 0.4 * base_reward + 0.6 * alignment + np.random.normal(0, 0.05)
    return float(np.clip(reward, 0.0, 1.0))


def _estimate_cp_loss(board: chess.Board, move: chess.Move) -> float:
    """Estimate cp loss heuristically without Stockfish."""
    # Simple heuristic based on move characteristics
    loss = np.random.exponential(50)  # Base random loss

    if board.is_capture(move):
        loss *= 0.7  # Captures tend to be reasonable
    if board.gives_check(move):
        loss *= 0.5  # Checks tend to be reasonable

    # Penalize moves to edge
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    if to_file in [0, 7] or to_rank in [0, 7]:
        loss *= 1.3

    return float(loss)


def _generate_positions(n: int = 50) -> list[chess.Board]:
    """Generate a set of varied positions for teaching."""
    positions = []

    # Common opening positions
    openings = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
        "r1bqkb1r/pppppppp/2n5/4P3/8/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 3",
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ]

    # Middlegame positions
    middlegames = [
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",
        "r2qkb1r/pp2pppp/2n2n2/3p1b2/3P4/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 4 6",
        "r1bqr1k1/ppp2ppp/2nb1n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
    ]

    # Endgame positions
    endgames = [
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        "8/5k2/8/8/3K4/8/3R4/8 w - - 0 1",
        "8/8/3k4/8/3KP3/8/8/8 w - - 0 1",
    ]

    all_fens = openings + middlegames + endgames
    for fen in all_fens:
        positions.append(chess.Board(fen))

    # Duplicate to fill
    while len(positions) < n:
        positions.append(positions[np.random.randint(len(all_fens))].copy())

    return positions[:n]


def run_experiment(
    students,
    policies: dict,
    n_episodes: int = 100,
    n_interactions_per_episode: int = 20,
    feedback_generator=None,
    context_builder_fn=None,
    reward_fn=None,
    positions: list[chess.Board] | None = None,
) -> dict:
    """Run full experiment comparing multiple policies."""
    from ..teaching.context import build_context
    from ..teaching.reward import compute_reward as default_reward
    from ..feedback.generator import FeedbackGenerator

    if feedback_generator is None:
        feedback_generator = FeedbackGenerator()
    if context_builder_fn is None:
        context_builder_fn = build_context
    if reward_fn is None:
        reward_fn = default_reward
    if positions is None:
        positions = _generate_positions(n_interactions_per_episode * 2)

    results = {}

    for policy_name, policy_template in policies.items():
        print(f"Running policy: {policy_name}")
        all_rewards = []
        all_elo_gains = []
        all_arm_counts = []
        all_regret_curves = []

        # Policy persists across episodes to accumulate learning
        policy = copy.deepcopy(policy_template)

        for ep in range(n_episodes):
            student_idx = ep % len(students)
            student = copy.deepcopy(students[student_idx])

            initial_elo = student.elo

            episode_result = run_episode(
                student=student,
                policy=policy,
                feedback_generator=feedback_generator,
                context_builder_fn=context_builder_fn,
                reward_fn=reward_fn,
                positions=positions,
                n_interactions=n_interactions_per_episode,
            )

            all_rewards.append(episode_result["cumulative_reward"])
            all_elo_gains.append(student.elo - initial_elo)
            all_arm_counts.append(np.bincount(
                episode_result["arms_selected"], minlength=7
            ))

            # Regret curve
            oracle_reward = 1.0
            cum_regret = np.cumsum(
                [oracle_reward - r for r in episode_result["rewards"]]
            )
            all_regret_curves.append(cum_regret)

        # Pad regret curves to same length
        max_len = max(len(rc) for rc in all_regret_curves)
        padded = []
        for rc in all_regret_curves:
            if len(rc) < max_len:
                rc = np.pad(rc, (0, max_len - len(rc)), constant_values=rc[-1] if len(rc) > 0 else 0)
            padded.append(rc)

        results[policy_name] = {
            "mean_cumulative_reward": float(np.mean(all_rewards)),
            "std_cumulative_reward": float(np.std(all_rewards)),
            "mean_elo_gain": float(np.mean(all_elo_gains)),
            "std_elo_gain": float(np.std(all_elo_gains)),
            "regret_curves": np.array(padded),
            "arm_distribution": np.mean(all_arm_counts, axis=0),
        }

    return results
