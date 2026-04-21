"""Feedback templates organized by type and ELO range."""

from .taxonomy import FeedbackType

# Templates keyed by (FeedbackType, elo_tier) where elo_tier is 'beginner', 'intermediate', 'advanced'
TEMPLATES: dict[FeedbackType, dict[str, list[str]]] = {
    FeedbackType.TACTICAL_ALERT: {
        "beginner": [
            "Look carefully at the board — there might be a way to win material! Check if any of your pieces can capture something valuable.",
            "Hint: one of your pieces can attack two things at once. Can you find it?",
            "There's a tactic here! Look at what happens if you move your {piece} to {square}.",
        ],
        "intermediate": [
            "There's a {tactic_type} available in this position. Look at the {square} square.",
            "This position has a tactical shot. Can you find the {tactic_type}?",
            "Consider the forcing move {piece} to {square} — it creates a {tactic_type}.",
        ],
        "advanced": [
            "A {tactic_type} motif exists here involving {piece}. The key square is {square}.",
            "The position demands concrete calculation — there's a {tactic_type} starting with {piece} to {square}.",
        ],
    },
    FeedbackType.STRATEGIC_NUDGE: {
        "beginner": [
            "Try to control the center of the board with your pawns and pieces.",
            "Think about developing your pieces — get your knights and bishops into the game!",
            "Your king might be safer after castling. Consider that plan.",
        ],
        "intermediate": [
            "This position calls for improving your worst-placed piece. Which piece is least active?",
            "Consider the pawn structure — {observation}. A plan involving {suggestion} could improve your position.",
            "The key strategic idea here is {concept}. Think about how to implement it.",
        ],
        "advanced": [
            "The critical strategic factor is {concept}. Consider {suggestion} to exploit the {observation}.",
            "Evaluate the long-term pawn structure implications. {observation} suggests {suggestion}.",
        ],
    },
    FeedbackType.BLUNDER_WARNING: {
        "beginner": [
            "Be careful! That move leaves your {piece} unprotected. Look for a safer option.",
            "Watch out — your opponent can capture your {piece} if you play that. Try again!",
            "That move looks natural, but it loses material. Can you see why?",
        ],
        "intermediate": [
            "Careful with that move — it loses {cp_loss} centipawns. {explanation}. Consider {alternative} instead.",
            "Watch out — that move allows {threat}. {alternative} is a better option here.",
            "That's a common mistake in this type of position. {explanation}.",
        ],
        "advanced": [
            "That move drops material due to {threat}. The engine line is {line}.",
            "Critical error: {explanation}. The refutation is {line}. Consider {alternative}.",
        ],
    },
    FeedbackType.PATTERN_RECOGNITION: {
        "beginner": [
            "This position looks like a common pattern! In positions like this, it's usually good to {suggestion}.",
            "Recognize this? It's similar to a basic checkmate pattern. Try to {suggestion}.",
        ],
        "intermediate": [
            "This position features a classic {pattern} pattern. The typical plan is {suggestion}.",
            "You've seen this type of position before — the key idea is {concept}.",
        ],
        "advanced": [
            "This is a well-known {pattern} structure. Theory suggests {suggestion}.",
            "The {pattern} motif here is similar to the classic game {reference}. Consider {suggestion}.",
        ],
    },
    FeedbackType.MOVE_COMPARISON: {
        "beginner": [
            "Let's compare: you played {student_move}. The engine suggests {engine_move}. Both are okay, but {explanation}.",
            "Your move was fine! The computer's top choice was {engine_move}, but your move is also reasonable.",
        ],
        "intermediate": [
            "You played {student_move} ({eval_student}). The engine prefers {engine_move} ({eval_engine}). The difference is {explanation}.",
            "Interesting choice! {student_move} vs {engine_move}: {explanation}.",
        ],
        "advanced": [
            "Your {student_move} ({eval_student}) vs the engine's {engine_move} ({eval_engine}): {explanation}. The critical line is {line}.",
        ],
    },
    FeedbackType.ENCOURAGEMENT: {
        "beginner": [
            "Great move! You're making good progress!",
            "Well done! That was a smart choice.",
            "Excellent! You found a really good move there.",
        ],
        "intermediate": [
            "Nice move! {student_move} was the engine's top choice too.",
            "Good instinct! That move improves your position by {reason}.",
            "Well played — you found the best move in a tricky position.",
        ],
        "advanced": [
            "Excellent technique! {student_move} is the engine's first choice.",
            "Strong move. You correctly identified the key feature of this position: {reason}.",
        ],
    },
    FeedbackType.SIMPLIFICATION: {
        "beginner": [
            "When you're ahead in material, try to trade pieces! Simpler positions are easier to win.",
            "You're doing well — consider exchanging some pieces to make the position clearer.",
        ],
        "intermediate": [
            "Consider simplifying with {suggestion}. Trading into a {endgame_type} gives you a clear advantage.",
            "The simplest path to victory is {suggestion}. Don't complicate things unnecessarily.",
        ],
        "advanced": [
            "The technique here is to convert via {suggestion}. The resulting {endgame_type} is winning because {explanation}.",
        ],
    },
}


def get_elo_tier(elo: int) -> str:
    """Map ELO to tier for template selection."""
    if elo < 1300:
        return "beginner"
    elif elo < 1700:
        return "intermediate"
    else:
        return "advanced"
