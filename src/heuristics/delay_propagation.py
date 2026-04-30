"""Route simulation and delay propagation."""
from __future__ import annotations

from src.env.entities import RouteRecord, Task, Worker
from src.heuristics.cost import route_distance, total_tardiness, travel_time


def simulate_route(
    worker: Worker,
    route: list[int],
    all_tasks: dict[int, Task],
    start_time: float,
    start_location: tuple[float, float] | None = None,
) -> list[RouteRecord]:
    """Simulate a committed route from the worker's current planning state.

    Returns one record per task containing arrival/start/finish/tardiness.
    Time windows are soft, so lateness does not make a route infeasible here.
    """
    time = max(float(start_time), float(worker.ready_time))
    loc = start_location if start_location is not None else worker.location
    records: list[RouteRecord] = []

    for task_id in route:
        task = all_tasks[task_id]
        arrival = time + travel_time(loc, task.location)
        start = max(arrival, task.earliest)
        finish = start + task.service_time
        tardiness = max(0.0, start - task.latest)
        records.append(
            RouteRecord(
                task_id=task_id,
                arrival_time=arrival,
                start_time=start,
                finish_time=finish,
                tardiness=tardiness,
            )
        )
        time = finish
        loc = task.location

    return records


def summarize_route(
    worker: Worker,
    route: list[int],
    all_tasks: dict[int, Task],
    start_time: float,
    start_location: tuple[float, float] | None = None,
) -> dict[str, float]:
    records = simulate_route(worker, route, all_tasks, start_time, start_location)
    return {
        "distance": route_distance(worker, route, all_tasks, start_location),
        "tardiness": total_tardiness(records),
        "finish_time": records[-1].finish_time if records else max(start_time, worker.ready_time),
    }
