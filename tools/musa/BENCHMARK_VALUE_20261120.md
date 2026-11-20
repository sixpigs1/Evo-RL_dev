# ValueFunction S5000 训练速度实测（2026-11-20）

## 配置与范围

- `pistar06`：预训练 SigLIP SO400M/14-384 + Qwen2.5-0.5B；融合层、201-bin 价值头按正常训练流程初始化。视觉/语言骨干均解冻，实际可训练参数 **923,937,929**。
- 原生 torch/torch_musa 2.9.0、Accelerate 1.15.0、DDP/MCCL，独立 `value` 环境；BF16、梯度检查点、GPU AdamW，无 CPU offload、无梯度累积、无 torch.compile；不保存 checkpoint、不上传实验平台、不改写原有权重。
- 单卡/两卡/四卡使用物理 GPU 2、2/3、2/3/4/5。每项开始前检查所需卡无活动计算任务，并按卡加锁，逐项串行运行；不停止其他进程。
- 相同公开毛巾验收子集：2 episodes、3,432 帧，真实成功标签 2 条；`legacy` 奖励/逐帧价值目标，三路 640×480 AV1 视频，PyAV，`num_workers=0`，无图像增强。标签复核见 `dataset.json`。
- 每项 35 个优化步，剔除第 1–10 步，统计第 11–35 步。端到端秒/步包括读取、预处理、更新和日志/计时收集开销；更新秒/步包括前向、反向、裁剪、优化器、调度器和 DDP 等待。GPU 在更新计时前后同步，多卡取各 rank 最大耗时。吞吐 = 全局 batch / 平均端到端秒/步。
- 每步用一次小张量 MCCL all-reduce 收集计时，带来少量开销。四卡 batch 32 会跨越数据集一轮；保持 Accelerate 默认尾批补齐，并记录 rank 0 实际目标张量的 batch 大小。数据循环重新读取，不使用预处理后的缓存 batch。
- 显存为各 rank 的 PyTorch peak allocated/reserved 最大值，不等于整卡占用。只做一次短测，记录标准差和 P90，不代表收敛、价值预测质量或长时稳定性。

## 结果

**6/6 组通过**。batch 均为每卡 batch，全局 batch = 每卡 batch × 卡数。

| 卡数 | 每卡 batch | 全局 batch | 端到端秒/步 | 更新秒/步 | 样本/秒 | 峰值 allocated / reserved GiB | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 16 | 3.157 | 1.116 | 5.068 | 11.251 / 13.555 | passed |
| 1 | 32 | 32 | 6.112 | 2.058 | 5.235 | 15.463 / 18.377 | passed |
| 2 | 16 | 32 | 3.322 | 1.239 | 9.634 | 12.983 / 14.977 | passed |
| 2 | 32 | 64 | 6.378 | 2.225 | 10.035 | 17.195 / 19.283 | passed |
| 4 | 16 | 64 | 3.423 | 1.249 | 18.695 | 12.982 / 14.977 | passed |
| 4 | 32 | 128 | 6.708 | 2.681 | 19.083 | 17.199 / 19.672 | passed |

### Batch 32 相对 batch 16 的吞吐差异

- 1 卡：5.07 → 5.24 样本/秒，+3.3%。
- 2 卡：9.63 → 10.03 样本/秒，+4.2%。
- 4 卡：18.70 → 19.08 样本/秒，+2.1%。

每卡 batch 相同的多卡对照会增大全局 batch。数据读取/预处理开销和共享服务器的调度会影响端到端吞吐，细小差值不宜视为确定提升。

## 复现和证据

```bash
cd /workspace/Evo-RL
EVORL_BENCH_ROOT=/workspace/storage/benchmarks/value-largebatch-new \
EVORL_BENCH_TASKS=value EVORL_BENCH_BATCHES="16 32" \
  bash tools/musa/benchmark_matrix.sh
python tools/musa/summarize_benchmark.py \
  /workspace/storage/benchmarks/value-largebatch-new --tasks value --batches 16 32
```

必须选择新的输出目录。单项入口 `benchmark.py --task value` 还支持显式数据/存储路径和每卡 batch；选择设备时设置 `MUSA_VISIBLE_DEVICES`，数量必须与 `--gpus` 相同。

日志、210 条逐步计时、CSV/JSON 汇总、标签检查、参数抽样同步检查、执行脚本快照与来源见 [`results/benchmark-20261120-value-largebatch/`](results/benchmark-20261120-value-largebatch/)。归档文本仅规范化行尾空白。服务器原始结果：`/data/yzw_explore/benchmarks/20261120-value-largebatch`，容器内 `/workspace/storage/benchmarks/20261120-value-largebatch`。环境锁定见 [`locks/value-py310.txt`](locks/value-py310.txt)，权重和数据 revision 见 [`sources.json`](sources.json)。
