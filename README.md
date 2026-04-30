# care_dispatch

动态照护调度仿真与训练项目。

## 数据

将 Solomon 数据集 `RC101.txt` 放到：

```bash
data/solomon/RC101.txt
```

正式实验建议在 `configs/env/rc101_default.yaml` 中保持：

```yaml
allow_synthetic_if_missing: false
```

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell 建议在项目根目录执行。若不安装 editable package，也可以临时使用：

```powershell
$env:PYTHONPATH="."
```

## 单元测试

```bash
pytest tests/
```

也支持直接运行单个测试文件，例如：

```bash
python tests/test_action_mask.py
```

## 训练 smoke test

先用很小步数确认脚本、状态、mask、PPO 更新链路能跑通：

```bash
python src/scripts/train_mlp_ppo.py --config configs/model/mlp_ppo.yaml --seed 0 --total-timesteps 64 --rollout-steps 16
python src/scripts/train_gat_ppo.py --config configs/model/gat_ppo.yaml --seed 0 --total-timesteps 64 --rollout-steps 16
```

当前版本在 PPO 内部使用：

```yaml
ppo:
  reward_scale: 0.002
```

这只缩放 PPO 的 advantage/return 训练目标，环境返回的 `reward = -delta_total_cost` 和所有成本日志不变，用于避免 critic 的 value loss 因成本量级过大而膨胀。

## 正式训练

```bash
python src/scripts/train_mlp_ppo.py --config configs/model/mlp_ppo.yaml --seed 0
python src/scripts/train_gat_ppo.py --config configs/model/gat_ppo.yaml --seed 0
```

输出位置：

```text
outputs/checkpoints/mlp_ppo_seed0.pt
outputs/checkpoints/gat_ppo_seed0.pt
outputs/logs/mlp_reward_curve.csv
outputs/logs/mlp_cost_curve.csv
outputs/logs/mlp_reject_curve.csv
outputs/logs/mlp_episode_metrics.csv
outputs/logs/gat_reward_curve.csv
outputs/logs/gat_cost_curve.csv
outputs/logs/gat_reject_curve.csv
outputs/logs/gat_episode_metrics.csv
```

## 小规模正确性检验

```bash
python src/scripts/run_sanity_check.py --config configs/experiment/sanity_check.yaml
```

输出：

```text
outputs/tables/sanity_check_results.csv
outputs/figures/sanity_gap.png
outputs/figures/sanity_runtime.png
```

`solver: gurobi` 会优先尝试 Gurobi；如果本机没有 Gurobi，会自动回退到 OR-Tools Routing 高质量基准。若 OR-Tools 未安装，请运行：

```bash
pip install ortools
```

## GAT vs MLP 消融

如果前面已经跑过 64 步 smoke test，目录中会存在短训练 checkpoint。为了避免消融误加载这些短训练模型，可以使用 `--force-train` 覆盖重训。

单种子、小步数消融 smoke test：

```bash
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml --seeds 0 --total-timesteps 512 --rollout-steps 128 --eval-episodes 3 --force-train
```

三种子、较短正式预跑：

```bash
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml --seeds 0,1,2 --total-timesteps 5000 --rollout-steps 512 --eval-episodes 5 --force-train
```

手册默认正式命令：

```bash
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml
```

输出：

```text
outputs/tables/gat_vs_mlp_results.csv
outputs/figures/gat_vs_mlp_reward_curve.png
outputs/figures/gat_vs_mlp_cost_curve.png
```
