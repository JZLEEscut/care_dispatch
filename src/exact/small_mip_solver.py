"""Small-instance exact / high-quality baseline solver.

The public function follows the interface required by the development manual:

    solve_small_instance_exact_or_baseline(instance, solver='gurobi') -> dict

For this milestone, Gurobi is not required.  If solver='gurobi' is requested but
Gurobi is unavailable, the function automatically falls back to OR-Tools
Routing.  If OR-Tools is also unavailable, the return value is feasible=False
with a clear reason instead of crashing.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any

import math

from src.env.entities import Task, Worker
from src.heuristics.cost import distance, get_cost_value, route_distance, tardiness_weight_for_task
from src.heuristics.delay_propagation import simulate_route
from src.heuristics.insertion import evaluate_worker_insertion


def compute_routes_objective(
    routes: list[list[int]],
    workers: list[Worker],
    tasks: dict[int, Task],
    config: dict,
    rejected_task_ids: list[int] | None = None,
) -> float:
    total = 0.0
    distance_weight = get_cost_value(config, 'distance_weight', 1.0)
    for worker, route in zip(workers, routes):
        total += distance_weight * route_distance(worker, route, tasks, start_location=worker.location)
        records = simulate_route(worker, route, tasks, start_time=worker.shift_start, start_location=worker.location)
        for rec in records:
            total += tardiness_weight_for_task(tasks[rec.task_id], config) * rec.tardiness
    for task_id in rejected_task_ids or []:
        task = tasks[task_id]
        if task.priority == 1:
            total += get_cost_value(config, 'reject_penalty_emergency', 200.0)
        else:
            total += get_cost_value(config, 'reject_penalty_regular', 80.0)
    return float(total)


def validate_routes(routes: list[list[int]], workers: list[Worker], tasks: dict[int, Task]) -> dict[str, int]:
    skill_violations = 0
    time_errors = 0
    duplicate_count = 0
    seen: set[int] = set()
    for worker, route in zip(workers, routes):
        for task_id in route:
            if task_id in seen:
                duplicate_count += 1
            seen.add(task_id)
            if worker.skill < tasks[task_id].skill_req:
                skill_violations += 1
        records = simulate_route(worker, route, tasks, start_time=worker.shift_start, start_location=worker.location)
        last_finish = worker.shift_start
        for rec in records:
            task = tasks[rec.task_id]
            expected_tardiness = max(0.0, rec.start_time - task.latest)
            if rec.start_time + 1e-7 < task.earliest:
                time_errors += 1
            if abs(rec.tardiness - expected_tardiness) > 1e-6:
                time_errors += 1
            if rec.finish_time + 1e-7 < rec.start_time or rec.finish_time + 1e-7 < last_finish:
                time_errors += 1
            last_finish = rec.finish_time
    return {
        'skill_violation_count': int(skill_violations),
        'time_calculation_error_count': int(time_errors),
        'duplicate_task_count': int(duplicate_count),
    }


def greedy_insertion_solution(instance: dict[str, Any], allow_reject: bool = False) -> dict[str, Any]:
    start = perf_counter()
    config = instance['config']
    workers = deepcopy(instance['workers'])
    tasks = {tid: deepcopy(t) for tid, t in instance['tasks'].items()}
    routes = [w.route for w in workers]
    rejected: list[int] = []

    ordered_tasks = sorted(tasks.values(), key=lambda t: (t.release_time, -t.priority, t.latest, t.id))
    for task in ordered_tasks:
        best_worker = None
        best_result = None
        now = float(task.release_time)
        for i, worker in enumerate(workers):
            result = evaluate_worker_insertion(worker, task, tasks, now, config)
            if result.feasible and (best_result is None or result.delta_total_cost < best_result.delta_total_cost):
                best_worker = i
                best_result = result
        if best_result is None or best_worker is None:
            if allow_reject:
                task.status = 'rejected'
                rejected.append(task.id)
            else:
                return {
                    'objective': math.inf,
                    'runtime': perf_counter() - start,
                    'feasible': False,
                    'routes': [list(w.route) for w in workers],
                    'rejected_task_ids': rejected,
                    'reason': 'greedy_no_feasible_insertion',
                }
        else:
            workers[best_worker].route = list(best_result.new_route)
            task.status = 'assigned'

    routes = [list(w.route) for w in workers]
    objective = compute_routes_objective(routes, workers, tasks, config, rejected)
    checks = validate_routes(routes, workers, tasks)
    return {
        'objective': objective,
        'runtime': perf_counter() - start,
        'feasible': checks['skill_violation_count'] == 0 and checks['duplicate_task_count'] == 0,
        'routes': routes,
        'rejected_task_ids': rejected,
        'reason': 'ok',
        **checks,
    }


def _try_gurobi(_instance: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return None
    # The project supports the required interface, but the Gurobi model is not
    # implemented in this compact milestone because OR-Tools is the intended
    # license-free baseline path for the current stage.
    return None


def _solve_with_ortools(instance: dict[str, Any], time_limit_seconds: int = 10) -> dict[str, Any]:
    start_clock = perf_counter()
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as exc:
        return {
            'objective': math.nan,
            'runtime': perf_counter() - start_clock,
            'feasible': False,
            'routes': [],
            'reason': f'ortools_unavailable: {exc}',
        }

    tasks = instance['tasks']
    workers = instance['workers']
    config = instance['config']
    task_ids = sorted(tasks)
    n_tasks = len(task_ids)
    n_workers = len(workers)
    start_nodes = list(range(n_tasks, n_tasks + n_workers))
    end_nodes = list(range(n_tasks + n_workers, n_tasks + 2 * n_workers))
    num_nodes = n_tasks + 2 * n_workers
    node_to_task_id = {i: task_ids[i] for i in range(n_tasks)}
    end_node_set = set(end_nodes)

    # Node locations: task nodes first, then per-worker start/end nodes.
    locations: list[tuple[float, float]] = []
    for tid in task_ids:
        locations.append(tasks[tid].location)
    for w in workers:
        locations.append(w.location)
    for w in workers:
        locations.append(w.location)

    manager = pywrapcp.RoutingIndexManager(num_nodes, n_workers, start_nodes, end_nodes)
    routing = pywrapcp.RoutingModel(manager)

    def _dist_units(a: tuple[float, float], b: tuple[float, float]) -> int:
        return int(round(distance(a, b)))

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node in end_node_set:
            return 0
        return _dist_units(locations[from_node], locations[to_node])

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        service = 0
        if from_node in node_to_task_id:
            service = int(round(tasks[node_to_task_id[from_node]].service_time))
        if to_node in end_node_set:
            return service
        return service + _dist_units(locations[from_node], locations[to_node])

    dist_cb = routing.RegisterTransitCallback(distance_callback)
    time_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)

    horizon = int(max(w.shift_end for w in workers)) + 10_000
    routing.AddDimension(time_cb, 10_000, horizon, False, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')

    for v, worker in enumerate(workers):
        start_idx = routing.Start(v)
        time_dim.CumulVar(start_idx).SetRange(int(round(worker.shift_start)), int(round(worker.shift_end)))
        end_idx = routing.End(v)
        time_dim.CumulVar(end_idx).SetRange(int(round(worker.shift_start)), int(round(worker.shift_end)))

    for node, task_id in node_to_task_id.items():
        idx = manager.NodeToIndex(node)
        task = tasks[task_id]
        time_dim.CumulVar(idx).SetMin(int(round(task.earliest)))
        latest = int(round(task.latest))
        coeff = max(1, int(round(tardiness_weight_for_task(task, config))))
        time_dim.SetCumulVarSoftUpperBound(idx, latest, coeff)
        allowed = [i for i, w in enumerate(workers) if w.skill >= task.skill_req]
        if not allowed:
            return {
                'objective': math.inf,
                'runtime': perf_counter() - start_clock,
                'feasible': False,
                'routes': [[] for _ in workers],
                'reason': f'no_skill_feasible_worker_for_task_{task_id}',
            }
        routing.SetAllowedVehiclesForIndex(allowed, idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_seconds))
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {
            'objective': math.inf,
            'runtime': perf_counter() - start_clock,
            'feasible': False,
            'routes': [[] for _ in workers],
            'reason': 'ortools_no_solution',
        }

    routes: list[list[int]] = []
    for v in range(n_workers):
        route: list[int] = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node in node_to_task_id:
                route.append(node_to_task_id[node])
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)

    objective = compute_routes_objective(routes, workers, tasks, config)
    checks = validate_routes(routes, workers, tasks)
    feasible = checks['skill_violation_count'] == 0 and checks['duplicate_task_count'] == 0
    return {
        'objective': objective,
        'runtime': perf_counter() - start_clock,
        'feasible': bool(feasible),
        'routes': routes,
        'reason': 'ok',
        **checks,
    }


def solve_small_instance_exact_or_baseline(instance: dict[str, Any], solver: str = 'gurobi') -> dict[str, Any]:
    solver = str(solver).lower()
    if solver == 'gurobi':
        result = _try_gurobi(instance)
        if result is not None:
            return result
        return _solve_with_ortools(instance)
    if solver in {'ortools', 'or-tools', 'routing'}:
        return _solve_with_ortools(instance)
    if solver in {'greedy', 'heuristic'}:
        return greedy_insertion_solution(instance, allow_reject=False)
    return {
        'objective': math.nan,
        'runtime': 0.0,
        'feasible': False,
        'routes': [],
        'reason': f'unknown_solver_{solver}',
    }
