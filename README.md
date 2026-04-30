# care_dispatch

动态照护调度仿真与训练项目。当前版本已按两份开发手册推进到实验二、实验三剩余消融和实验四脚本阶段：

1. 事件驱动动态照护调度环境。
2. 底层插入启发式与有限抢占候选。
3. GAT 图状态构造与 MLP 扁平状态构造。
4. GAT-PPO-Heuristic 与 MLP-PPO-Heuristic 训练入口。
5. 小规模正确性检验脚本。
6. GAT 表征消融脚本。
7. 实验二综合性能对比：Pure Heuristic、Rolling OR-Tools、Rolling ALNS、MLP-PPO、GAT-PPO。
8. 实验三剩余消融：Action Masking 消融、抢占机制消融。
9. 实验四零样本空间泛化：RC101 训练，RC101/C101/R101 测试。

仍未加入管理启示、参数敏感性、压力测试、多智能体 RL、DQN/DDPG/SAC、2M+1 动作空间或“任务-护工-插入位置”联合动作空间。

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

Windows PowerShell 建议在项目根目录执行。若你不安装 editable package，也可以临时使用：

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

## 建议开发顺序

1. 先运行 `pytest tests/`。
2. 再运行 MLP/GAT smoke test。
3. 运行 `run_sanity_check.py` 检查 `skill_violation_count` 和 `time_calculation_error_count` 是否为 0。
4. 运行单种子、小步数 `run_gat_ablation.py --force-train`，确认消融链路能覆盖训练并生成结果表。
5. 链路稳定后再扩展到 `--seeds 0,1,2` 和更大的训练步数。

## Milestone 5 notes: avoiding the all-reject early policy

If `run_gat_ablation.py` reports `reject_rate=1.0` for both GAT and MLP after a short run, retrain from scratch with the updated model configuration:

```bash
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml --seeds 0 --total-timesteps 1000 --rollout-steps 128 --eval-episodes 3 --force-train
```

This version keeps the required `M+1` action space and keeps reject always legal, but adds two stabilizers:

1. `action_mask.feasibility_check: true` masks workers that the bottom-level insertion heuristic already knows are infeasible for the current task.
2. `reject_init_bias: -3.0` slightly discourages the always-legal reject action at initialization. PPO can still learn reject whenever it is cost-effective.

For a longer pre-experiment:

```bash
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml --seeds 0,1,2 --total-timesteps 5000 --rollout-steps 512 --eval-episodes 5 --force-train
```


## 里程碑 6 说明

本版本修复了 `mean_emergency_response_time` 始终为 0 的评价口径问题。
紧急任务响应时间现在按“任务 release_time 到计划服务 start_time”的差值统计，只影响日志与评价指标，不改变环境奖励函数。

建议先运行：

```bash
pytest tests/
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml --seeds 0,1,2 --total-timesteps 5000 --rollout-steps 512 --eval-episodes 5 --force-train
```

## 里程碑 7 说明：RC101 拒单惩罚尺度校准

在 RC101 的紧时间窗数据上，`reject_penalty_regular=80` 与 `reject_penalty_emergency=200` 会使“几乎全拒单”的累计成本低于大量服务但产生高迟到的方案，长训练后 PPO 会理性收敛到高拒单策略。为满足“reject_count 不应长期等于任务总数”的训练成功标准，本版本只调整成本参数尺度，不改变奖励公式、动作空间或底层启发式。

当前默认值：

```yaml
cost:
  reject_penalty_regular: 500.0
  reject_penalty_emergency: 1500.0
```

由于拒单惩罚变大，PPO 内部缩放调整为：

```yaml
ppo:
  reward_scale: 0.002
```

如果需要临时覆盖成本尺度，可以在训练或消融命令中加入：

```bash
python src/scripts/run_gat_ablation.py --config configs/experiment/gat_ablation.yaml --seeds 0 --total-timesteps 20000 --rollout-steps 1024 --eval-episodes 10 --force-train --reject-penalty-regular 500 --reject-penalty-emergency 1500 --reward-scale 0.002
```

## 重新绘制平滑训练曲线

PPO 的 episode 级 reward/cost 天然波动较大。消融脚本默认会输出 rolling mean 曲线，同时保留 `*_raw.png` 原始曲线副本。已有日志时，可以不重新训练，直接重绘：

```bash
python src/scripts/plot_training_curves.py --plot-window 100
```

若需要完全原始曲线：

```bash
python src/scripts/plot_training_curves.py --plot-window 1
```


## 实验二、三、四

实验二：综合性能对比实验。

```bash
python src/scripts/run_benchmark.py --config configs/experiment/benchmark.yaml
```

调试时建议先缩小规模：

```bash
python src/scripts/run_benchmark.py --config configs/experiment/benchmark.yaml --seeds 0 --eval-episodes 2 --total-timesteps 1024 --rollout-steps 128 --force-train
```

训练 Action Masking 消融模型：

```bash
python src/scripts/train_gat_ppo.py --config configs/model/gat_ppo_nomask.yaml --seed 0
```

训练抢占机制消融模型：

```bash
python src/scripts/train_gat_ppo.py --config configs/model/gat_ppo_nopreempt.yaml --seed 0
```

实验三：剩余消融实验。

```bash
python src/scripts/run_remaining_ablation.py --config configs/experiment/remaining_ablation.yaml
```

实验四：零样本空间泛化实验。请先把 `C101.txt` 和 `R101.txt` 放到 `data/solomon/`。脚本不会在 C101/R101 上重新训练或微调，只加载 RC101 checkpoint。

```bash
python src/scripts/run_zero_shot_spatial.py --config configs/experiment/zero_shot_spatial.yaml
```

如果只是检查脚本链路且暂时没有 C101/R101，可使用：

```bash
python src/scripts/run_zero_shot_spatial.py --config configs/experiment/zero_shot_spatial.yaml --skip-missing --seeds 0 --eval-episodes 2
```
