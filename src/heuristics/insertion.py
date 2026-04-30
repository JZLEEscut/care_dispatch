"""Worker-level insertion heuristic."""
from __future__ import annotations

from dataclasses import replace

from src.env.entities import InsertionResult, Task, Worker
from src.heuristics.cost import (
    get_cost_value,
    route_distance,
    tardiness_weight_for_task,
    total_tardiness,
)
from src.heuristics.delay_propagation import simulate_route
from src.heuristics.legality import route_within_shift, shift_available, skill_feasible
from src.heuristics.preemption import can_consider_preemption, preemptive_route


def _route_cost_parts(worker: Worker, route: list[int], all_tasks: dict[int, Task], now: float) -> tuple[float, float, float]:
    records = simulate_route(worker, route, all_tasks, start_time=now, start_location=worker.location)
    dist = route_distance(worker, route, all_tasks, start_location=worker.location)
    tard = total_tardiness(records)
    finish = records[-1].finish_time if records else max(now, worker.ready_time)
    return dist, tard, finish


def _candidate_result(
    worker: Worker,
    task: Task,
    all_tasks: dict[int, Task],
    now: float,
    config,
    candidate_route: list[int],
    old_distance: float,
    old_tardiness: float,
    preempted: bool,
) -> InsertionResult:
    new_distance, new_tardiness, finish_time = _route_cost_parts(worker, candidate_route, all_tasks, now)
    if not route_within_shift(finish_time, worker):
        return InsertionResult(False, list(worker.route), reason="finish_after_shift_end")

    delta_distance = new_distance - old_distance
    delta_tardiness = new_tardiness - old_tardiness
    delta_preemption = 1.0 if preempted else 0.0

    total = (
        get_cost_value(config, "distance_weight", 1.0) * delta_distance
        + tardiness_weight_for_task(task, config) * delta_tardiness
        + get_cost_value(config, "mu_preemption", 30.0) * delta_preemption
    )
    return InsertionResult(
        feasible=True,
        new_route=list(candidate_route),
        delta_distance_cost=float(delta_distance),
        delta_tardiness_cost=float(delta_tardiness),
        delta_preemption_cost=float(delta_preemption),
        delta_total_cost=float(total),
        preempted=preempted,
        reason="ok",
    )


def _dedupe_preserving_order(route: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for t in route:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def evaluate_worker_insertion(
    worker: Worker,
    task: Task,
    all_tasks: dict[int, Task],
    now: float,
    config,
) -> InsertionResult:
    """Evaluate all legal insertion positions and optional preemption.

    The upper-level action only chooses a worker. This function checks hard
    constraints, enumerates insertion positions, optionally adds finite
    preemption candidates for emergency tasks, and returns the minimum-cost
    feasible route update.
    """
    if not skill_feasible(worker, task):
        return InsertionResult(False, list(worker.route), reason="skill_infeasible")
    if not shift_available(worker, now):
        return InsertionResult(False, list(worker.route), reason="worker_unavailable_after_shift")
    if task.id in worker.route:
        return InsertionResult(False, list(worker.route), reason="task_already_in_route")

    old_distance, old_tardiness, _ = _route_cost_parts(worker, list(worker.route), all_tasks, now)

    best: InsertionResult | None = None
    route = list(worker.route)
    route_len = len(route)

    # If the worker is already committed to the first route task (traveling or
    # serving), normal insertion may only happen after that task. Placing an
    # emergency before it must pay the preemption cost and pass preemption rules.
    normal_start_pos = 0
    if worker.current_status in {"traveling", "serving"} and route:
        normal_start_pos = 1

    for pos in range(normal_start_pos, route_len + 1):
        candidate = route[:pos] + [task.id] + route[pos:]
        candidate = _dedupe_preserving_order(candidate)
        result = _candidate_result(
            worker, task, all_tasks, now, config, candidate, old_distance, old_tardiness, preempted=False
        )
        if result.feasible and (best is None or result.delta_total_cost < best.delta_total_cost):
            best = result

    ok_preempt, _ = can_consider_preemption(worker, task, all_tasks, config)
    if ok_preempt:
        candidate = _dedupe_preserving_order(preemptive_route(worker, task))
        result = _candidate_result(
            worker, task, all_tasks, now, config, candidate, old_distance, old_tardiness, preempted=True
        )
        if result.feasible and (best is None or result.delta_total_cost < best.delta_total_cost):
            best = result

    if best is None:
        return InsertionResult(False, list(worker.route), reason="no_feasible_insertion")
    return best
