"""Episode metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EpisodeMetrics:
    episode_reward: float = 0.0
    total_cost: float = 0.0
    distance_cost: float = 0.0
    tardiness_cost: float = 0.0
    preemption_count: int = 0
    reject_count: int = 0
    reject_cost: float = 0.0
    preemption_cost: float = 0.0
    emergency_reject_count: int = 0
    illegal_action_count: int = 0
    illegal_skill_mismatch_count: int = 0
    illegal_overtime_count: int = 0
    illegal_preemption_count: int = 0
    action_mask_all_zero_count: int = 0
    decision_time_total: float = 0.0
    decision_count: int = 0
    emergency_response_time_total: float = 0.0
    emergency_response_count: int = 0

    @property
    def mean_decision_time(self) -> float:
        if self.decision_count == 0:
            return 0.0
        return self.decision_time_total / self.decision_count

    @property
    def mean_emergency_response_time(self) -> float:
        if self.emergency_response_count == 0:
            return 0.0
        return self.emergency_response_time_total / self.emergency_response_count

    def as_dict(self) -> dict:
        d = asdict(self)
        d['mean_decision_time'] = self.mean_decision_time
        d['mean_emergency_response_time'] = self.mean_emergency_response_time
        return d
