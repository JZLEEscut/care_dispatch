from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.dispatch_env import CareDispatchEnv
from src.scripts._script_utils import load_yaml


def test_nomask_illegal_action_advances_environment():
    cfg = load_yaml('configs/env/rc101_default.yaml')
    cfg['data']['num_tasks'] = 5
    cfg['data']['num_workers'] = 2
    cfg['skills']['worker_skill_distribution'] = [1.0, 0.0, 0.0]
    cfg['skills']['task_skill_distribution'] = [0.0, 0.0, 1.0]
    cfg['action_mask'] = {'enabled': False, 'feasibility_check': True}
    env = CareDispatchEnv(cfg)
    state = env.reset(seed=0)
    first_id = env.active_task.id
    next_state, reward, done, info = env.step(0)
    assert info['illegal_action'] is True
    assert reward < 0
    assert env.tasks[first_id].status == 'rejected_illegal'
    assert done or env.active_task is None or env.active_task.id != first_id
