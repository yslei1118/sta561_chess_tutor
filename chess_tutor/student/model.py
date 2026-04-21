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
        """Classify recent trajectory as improving / stable / declining.

        Uses a two-window Welch t-statistic on the last 30 cp_losses (15
        recent, 15 older) and only declares a trend when |t| > 1.5 (≈90%
        one-sided confidence).  This replaces the old ±20%-of-mean rule,
        which flipped erratically under the high variance of per-move
        cp_loss (a single blunder could swing a 10-move window's mean by
        2×).  Requiring 30 samples + a t-test keeps the trend field
        informative for downstream feedback templates.
        """
        if len(self.recent_cp_losses) < 30:
            self.trend = "stable"
            return

        recent = np.array(self.recent_cp_losses[-15:], dtype=float)
        older = np.array(self.recent_cp_losses[-30:-15], dtype=float)

        mean_diff = older.mean() - recent.mean()  # positive = improvement
        # Welch's approximation with equal sample size n=15 per window.
        var_sum = recent.var(ddof=1) + older.var(ddof=1)
        if var_sum < 1e-6:
            self.trend = "stable"
            return
        t_stat = mean_diff / np.sqrt(var_sum / 15.0)

        if t_stat > 1.5:
            self.trend = "improving"
        elif t_stat < -1.5:
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
