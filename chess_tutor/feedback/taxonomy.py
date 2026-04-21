"""Feedback type taxonomy and concept mappings."""

from enum import IntEnum


class FeedbackType(IntEnum):
    TACTICAL_ALERT = 0       # F1: Tactic available
    STRATEGIC_NUDGE = 1      # F2: Positional improvement
    BLUNDER_WARNING = 2      # F3: High blunder probability
    PATTERN_RECOGNITION = 3  # F4: Known pattern match
    MOVE_COMPARISON = 4      # F5: Compare with engine
    ENCOURAGEMENT = 5        # F6: Good move praise
    SIMPLIFICATION = 6       # F7: Suggest simplifying


FEEDBACK_CONCEPT_MAP: dict[FeedbackType, list[str]] = {
    FeedbackType.TACTICAL_ALERT: ["tactics"],
    FeedbackType.STRATEGIC_NUDGE: ["strategy", "positional"],
    FeedbackType.BLUNDER_WARNING: ["tactics", "calculation"],
    FeedbackType.PATTERN_RECOGNITION: ["opening", "endgame", "pattern"],
    FeedbackType.MOVE_COMPARISON: ["calculation", "strategy"],
    FeedbackType.ENCOURAGEMENT: [],
    FeedbackType.SIMPLIFICATION: ["endgame", "strategy"],
}
