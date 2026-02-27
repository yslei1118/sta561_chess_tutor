"""Student state model."""

from dataclasses import dataclass, field
import numpy as np

from ..config import STUDENT_BASE_CONCEPTS


@dataclass
class StudentState:
    """Represents the current state of a student."""
    elo: int = 1200
    weakness_profile: dict[str, float] = field(default_factory=lambda: {
        c: 0.5 for c in STUDENT_BASE_CONCEPTS
    })
    recent_cp_losses: list[float] = field(default_factory=list)
    blunder_count: int = 0
    moves_played: int = 0
    trend: str = "stable"  # 'improving', 'stable', 'declining'

    def update_trend(self):
        """Update trend based on recent performance."""
        if len(self.recent_cp_losses) < 10:
            self.trend = "stable"
            return

        recent = self.recent_cp_losses[-10:]
        older = self.recent_cp_losses[-20:-10] if len(self.recent_cp_losses) >= 20 else self.recent_cp_losses[:10]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        if recent_avg < older_avg * 0.8:
            self.trend = "improving"
        elif recent_avg > older_avg * 1.2:
            self.trend = "declining"
        else:
            self.trend = "stable"

    def copy(self) -> "StudentState":
        return StudentState(
            elo=self.elo,
            weakness_profile=dict(self.weakness_profile),
            recent_cp_losses=list(self.recent_cp_losses),
            blunder_count=self.blunder_count,
            moves_played=self.moves_played,
            trend=self.trend,
        )
