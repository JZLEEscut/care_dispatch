from pathlib import Path
import sys
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scenario_builder import build_scenario
from src.env.entities import Task, Worker
from src.heuristics.insertion import evaluate_worker_insertion
from src.scripts._script_utils import load_yaml


def make_busy_worker_and_emergency():
    regular = Task(0, 10, 0, 10, 0, 200, 1, 0, True, 0, False, 'assigned')
    emergency = Task(1, 1, 0, 10, 0, 200, 1, 1, False, 5, True, 'waiting')
    worker = Worker(0, 3, 0, 0, 0, 480, 5, route=[0], current_task_id=0, current_status='traveling', current_task_interruptible=True)
    return worker, emergency, {0: regular, 1: emergency}


def test_preemption_toggle_controls_candidate():
    cfg = load_yaml('configs/env/rc101_default.yaml')
    worker, emergency, tasks = make_busy_worker_and_emergency()
    cfg['preemption'] = {'enabled': True}
    res_on = evaluate_worker_insertion(worker, emergency, tasks, 5.0, cfg)
    assert res_on.feasible
    cfg['preemption'] = {'enabled': False}
    res_off = evaluate_worker_insertion(worker, emergency, tasks, 5.0, cfg)
    assert res_off.delta_preemption_cost == 0.0
    assert res_off.preempted is False
