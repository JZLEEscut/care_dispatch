from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scenario_builder import build_scenario
from src.graph.graph_builder import build_graph_state, NODE_FEATURE_DIM
from src.env.state_builder import build_state_dict
from src.scripts._script_utils import load_yaml


def test_zero_shot_config_can_switch_solomon_file_if_present():
    cfg = load_yaml('configs/env/rc101_default.yaml')
    cfg['data']['num_tasks'] = 5
    cfg['data']['num_workers'] = 2
    for fname in ['RC101.txt', 'C101.txt', 'R101.txt']:
        path = PROJECT_ROOT / 'data' / 'solomon' / fname
        if not path.exists():
            if fname != 'RC101.txt':
                pytest.skip(f'{fname} not provided locally; zero-shot runner will require it for full experiment.')
        cfg['data']['solomon_file'] = f'data/solomon/{fname}'
        scenario = build_scenario(cfg, seed=0)
        state = build_state_dict(0.0, scenario['tasks'][0], scenario['workers'], {t.id: t for t in scenario['tasks']})
        graph = build_graph_state(state, [1.0] * (len(scenario['workers']) + 1), cfg)
        assert graph.node_features.shape[1] == NODE_FEATURE_DIM
