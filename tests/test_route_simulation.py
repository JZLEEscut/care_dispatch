from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.entities import Task, Worker
from src.heuristics.delay_propagation import simulate_route


def test_route_simulation_times_and_tardiness():
    worker = Worker(0, 3, 0, 0, 0, 480, 0)
    tasks = {
        0: Task(0, 3, 4, 10, 0, 20, 1, 0, True, 0, False),
        1: Task(1, 6, 8, 5, 20, 30, 1, 0, True, 0, False),
    }
    records = simulate_route(worker, [0, 1], tasks, 0)
    assert len(records) == 2
    prev_finish = 0.0
    for r in records:
        task = tasks[r.task_id]
        assert r.arrival_time >= prev_finish
        assert r.start_time >= task.earliest
        assert r.finish_time >= r.start_time
        assert r.tardiness == max(0.0, r.start_time - task.latest)
        prev_finish = r.finish_time
