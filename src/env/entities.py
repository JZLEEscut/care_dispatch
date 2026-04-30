"""Core entities for dynamic care dispatch.

This file intentionally contains only data containers and no solver logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    id: int
    x: float
    y: float
    service_time: float
    earliest: float
    latest: float
    skill_req: int
    priority: int  # 0=regular, 1=emergency
    interruptible: bool
    release_time: float
    is_dynamic: bool
    status: str = "waiting"  # waiting/assigned/in_service/done/rejected

    @property
    def location(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Worker:
    id: int
    skill: int
    x: float
    y: float
    shift_start: float
    shift_end: float
    ready_time: float
    route: list[int] = field(default_factory=list)
    current_task_id: Optional[int] = None
    current_status: str = "idle"  # idle/traveling/serving
    current_task_interruptible: bool = False
    workload: float = 0.0

    @property
    def location(self) -> tuple[float, float]:
        return (self.x, self.y)

    def set_location(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


@dataclass
class RouteRecord:
    task_id: int
    arrival_time: float
    start_time: float
    finish_time: float
    tardiness: float


@dataclass
class InsertionResult:
    feasible: bool
    new_route: list[int]
    delta_distance_cost: float = 0.0
    delta_tardiness_cost: float = 0.0
    delta_preemption_cost: float = 0.0
    delta_total_cost: float = 0.0
    preempted: bool = False
    reason: str = ""


@dataclass
class StepResult:
    state: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]
