from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.entities import Task, Worker
from src.heuristics.insertion import evaluate_worker_insertion


def test_insertion_route_contains_task_no_duplicate_skill_ok():
    cfg = {"cost": {"distance_weight": 1.0, "lambda_tardiness_regular": 2.0, "mu_preemption": 30.0}}
    worker = Worker(0, 2, 0, 0, 0, 480, 0, route=[])
    task = Task(10, 1, 1, 5, 0, 100, 2, 0, True, 0, False)
    tasks = {10: task}
    result = evaluate_worker_insertion(worker, task, tasks, now=0, config=cfg)
    assert result.feasible
    assert result.new_route.count(10) == 1
    assert worker.skill >= task.skill_req


def test_insertion_rejects_skill_infeasible():
    worker = Worker(0, 1, 0, 0, 0, 480, 0, route=[])
    task = Task(10, 1, 1, 5, 0, 100, 3, 0, True, 0, False)
    result = evaluate_worker_insertion(worker, task, {10: task}, now=0, config={})
    assert not result.feasible
    assert result.reason == "skill_infeasible"
