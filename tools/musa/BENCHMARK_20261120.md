# S5000 训练速度实测（2026-11-20）

首轮覆盖 Evo-1 Stage 2 全量训练及 Pi0.5 全量微调，分别使用 1、2、4 张 S5000，每卡 batch 为 1、2、4、8。Pi0.5 的 batch 16、32 补测见文末。所有数值来自实际预训练模型、真实毛巾数据和优化器更新；不使用随机模型或合成输入。

## 实测结果

首轮 24 组中 **24 组通过**；每个通过项完成 35 步，稳态统计 25 步。通过项 loss 和梯度有限，多卡项结束后的参数抽样一致。以下 batch 均为**每卡 batch**，不是全局 batch。

### Evo-1 Stage 2 全量

| 卡数 | 每卡 batch | 全局 batch | 端到端秒/步 | 更新秒/步 | 全局样本/秒 | 峰值 allocated / reserved GiB | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 0.475 | 0.345 | 2.10 | 8.44 / 8.81 | passed |
| 1 | 2 | 2 | 0.727 | 0.477 | 2.75 | 8.44 / 9.37 | passed |
| 1 | 4 | 4 | 1.245 | 0.806 | 3.21 | 9.22 / 11.22 | passed |
| 1 | 8 | 8 | 1.975 | 1.494 | 4.05 | 12.25 / 13.80 | passed |
| 2 | 1 | 2 | 0.557 | 0.432 | 3.59 | 10.12 / 10.31 | passed |
| 2 | 2 | 4 | 0.843 | 0.601 | 4.74 | 10.12 / 11.44 | passed |
| 2 | 4 | 8 | 1.401 | 0.981 | 5.71 | 10.89 / 12.87 | passed |
| 2 | 8 | 16 | 2.258 | 1.737 | 7.09 | 13.93 / 15.04 | passed |
| 4 | 1 | 4 | 0.561 | 0.427 | 7.13 | 10.12 / 10.50 | passed |
| 4 | 2 | 8 | 0.835 | 0.565 | 9.59 | 10.12 / 11.43 | passed |
| 4 | 4 | 16 | 1.430 | 1.000 | 11.19 | 10.89 / 12.89 | passed |
| 4 | 8 | 32 | 2.387 | 1.831 | 13.41 | 13.93 / 15.71 | passed |

首轮 batch 1–8 范围内最高吞吐为 **4 卡、每卡 batch 8：13.41 样本/秒**。这不是对更大 batch 的上限搜索。

固定每卡 batch 8 时，相对单卡的 4.05 样本/秒，吞吐为 3.31 倍；按 4 卡线性扩展计算的效率为 82.7%。

### Pi0.5 全量

| 卡数 | 每卡 batch | 全局 batch | 端到端秒/步 | 更新秒/步 | 全局样本/秒 | 峰值 allocated / reserved GiB | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 0.831 | 0.771 | 1.20 | 32.13 / 32.54 | passed |
| 1 | 2 | 2 | 1.021 | 0.788 | 1.96 | 32.17 / 32.45 | passed |
| 1 | 4 | 4 | 1.409 | 0.930 | 2.84 | 32.17 / 32.50 | passed |
| 1 | 8 | 8 | 1.806 | 1.349 | 4.43 | 32.22 / 32.60 | passed |
| 2 | 1 | 2 | 1.189 | 1.123 | 1.68 | 39.12 / 39.52 | passed |
| 2 | 2 | 4 | 1.248 | 1.124 | 3.21 | 39.15 / 39.45 | passed |
| 2 | 4 | 8 | 1.827 | 1.302 | 4.38 | 39.17 / 39.47 | passed |
| 2 | 8 | 16 | 2.225 | 1.741 | 7.19 | 39.23 / 39.55 | passed |
| 4 | 1 | 4 | 1.044 | 0.970 | 3.83 | 39.12 / 39.52 | passed |
| 4 | 2 | 8 | 1.242 | 1.104 | 6.44 | 39.17 / 39.45 | passed |
| 4 | 4 | 16 | 1.662 | 1.397 | 9.63 | 39.14 / 39.47 | passed |
| 4 | 8 | 32 | 2.649 | 2.095 | 12.08 | 39.23 / 39.55 | passed |

首轮 batch 1–8 范围内最高吞吐为 **4 卡、每卡 batch 8：12.08 样本/秒**。这不是对更大 batch 的上限搜索。

固定每卡 batch 8 时，相对单卡的 4.43 样本/秒，吞吐为 2.73 倍；按 4 卡线性扩展计算的效率为 68.2%。

固定全局 batch 8 时，Pi0.5 单卡 batch 8 为 **4.43**、双卡每卡 batch 4 为 **4.38**、四卡每卡 batch 2 为 **6.44 样本/秒**。当前设置下，Pi0.5 双卡在相同总 batch 下没有明显吞吐收益，不能只依据每卡 batch 相同的扩展结果选择卡数。

### 与昨天 4090 的简单对照

Evo-1 使用相同**全局 batch 8**对照，均为 BF16、全量 Stage 2、开启 VLM 梯度检查点、关闭 FlashAttention，模型可训练参数均为 775,198,640：

| 配置 | 全局 batch | 端到端秒/步 | 更新秒/步 | 样本/秒 | 相对 4090 端到端吞吐 |
|---|---:|---:|---:|---:|---:|
| 昨天 RTX 4090，单卡 batch 8 | 8 | 1.472 | 1.361 | 5.43 | 1.00× |
| S5000 1 卡，每卡 batch 8 | 8 | 1.975 | 1.494 | 4.05 | 0.75× |
| S5000 2 卡，每卡 batch 4 | 8 | 1.401 | 0.981 | 5.71 | 1.05× |
| S5000 4 卡，每卡 batch 2 | 8 | 0.835 | 0.565 | 9.59 | 1.76× |

Pi0.5 本次确认全部 **3,616,757,520** 参数可训练。S5000 单卡 batch 1 的 GPU AdamW 全量训练更新均值为 **0.771 秒**，端到端 **0.831 秒**；4090 的历史全量 CPU offload 仅完成 3 步，更新均值 **10.58 秒**。这说明 S5000 的 80 GiB 显存可以支撑当前 GPU AdamW 全量方案，但两种优化器部署方式和测量长度不同，不把这个差值表述为 GPU 算力倍数；历史 LoRA 吞吐也不混入全量对照。

完整逐步数据、每项原始日志、实际执行的脚本和环境来源见 [`results/benchmark-20261120/`](results/benchmark-20261120/)。归档文本仅规范化行尾空白，训练指标未改动。`summary.json` / `summary.csv` 还包含标准差、P90、各 rank 显存及数值/同步检查。服务器原始结果保存在 `/data/yzw_explore/benchmarks/20261120`（容器内 `/workspace/storage/benchmarks/20261120`）。

## 测量方法

- 主机 CPU 为双路 AMD EPYC 9T34、内存约 2.2 TiB（见结果目录 `hardware.txt`）。容器 `yzw_explore`，原生 torch/torch_musa 2.9.0，Accelerate/DDP/MCCL；选择物理 GPU 2、3、4、5，逐项串行测试，避开 GPU 1 的其他任务。测试开始时 GPU 1 有其他计算任务，末段该任务已自行结束；本次没有停止任何其他进程。整机并非完全独占，背景负载变化也会影响对比。
- BF16，梯度检查点开启，Evo-1 关闭 FlashAttention，Pi0.5 关闭 torch.compile；梯度累积为 1，AdamW 优化器在 GPU 上更新。Evo-1 Stage 2 从真实 Stage 1 的第 100 步权重开始；Pi0.5 不使用 LoRA，视觉编码器和动作专家均可训练。
- 每组 35 个优化步，排除第 1–10 步，统计第 11–35 步。关闭评估、checkpoint 保存和实验平台上传；各项使用独立输出目录，不影响原有训练 checkpoint。
- `update_s`：同步设备后测量前向、反向、梯度裁剪、优化器和调度器更新，再同步设备；多卡取各 rank 的最大耗时。
- `step_s`：相邻优化步结束之间的间隔，包含读取真实视频、预处理、更新、日志和计时收集开销；多卡取每步最慢 rank。吞吐 = 全局 batch / 稳态平均 `step_s`，全局 batch = 每卡 batch × 卡数。
- 每步另有一次小张量 MCCL all-reduce 收集计时，会给端到端结果带来少量额外开销。结果为短测，无重复独立试验，不代表收敛或长时间稳定性。
- S5000 数据为 `Elvinky/bi_so101_fold_towel_d0` 的验收子集：2 episodes、3,432 帧、三路 640×480 AV1 视频，PyAV，`num_workers=0`，无图像增强。模型输入保持 Evo-1 448×448、动作块 50；Pi0.5 使用其预训练配置。
- 显存报告是所有 rank 中 PyTorch 峰值 allocated/reserved 的最大值，不等于整卡占用，也不与历史 `nvidia-smi` 整卡峰值直接比较。

## 4090 对照来源及限制

直接取回 4090 昨天（2026-11-19）Evo-1 Stage 2 原始日志，归档训练指标见 `results/baseline-4090-nov19-training.log`；剔除首个 100 步窗口后，71 个日志窗口的更新耗时均值为 1.360761 秒，数据耗时 0.111254 秒，共 1.472014 秒/步，batch 8 吞吐为 **5.434731 样本/秒**。该主机启用了 GPU Unlock；本次只读查询，没有改变它的驱动、启动设置或正在运行的训练。

11 月 18 日历史 batch sweep 的 Evo-1 Stage 2（每组 20 步，剔除前 5 步）还记录 batch 4/8/16 为 0.742/1.462/2.913 秒/步，即 5.39/5.47/5.49 样本/秒；batch 32 OOM。这些 `/tmp` 原始文件已随主机重启丢失，来源是先前 Codex 实测记录，单独保存在 `results/baseline-4090-history.json`，不能误称本次重测。

4090 使用用户的 40 条毛巾轨迹、47,398 帧；S5000 使用公开验收子集。两者均为三路相同分辨率视频，但 4090 使用 TorchCodec，S5000 使用 PyAV；数据内容、随机访问开销、CPU、存储、软件栈及文本长度也不完全一致。因此对照体现实际训练配置的速度，不能解释为纯 GPU 算力倍率。另列更新耗时以便观察数据加载的影响。

Pi0.5 历史 batch sweep 是 **LoRA rank 16**，不能与本次全量训练作等价速度对比。11 月 18 日历史全量训练仅有 batch 1 的逐张量 CPU AdamW offload 3 步短测，平均更新耗时 10.58 秒；默认 GPU AdamW 在 24 GiB 4090 上 OOM。CPU offload 与本次 GPU AdamW 的差异必须保留，不能据此宣称 GPU 本身的加速倍数。

## 复现

```bash
cd /workspace/Evo-RL
EVORL_BENCH_ROOT=/workspace/storage/benchmarks/new-run \
  bash tools/musa/benchmark_matrix.sh
python tools/musa/summarize_benchmark.py /workspace/storage/benchmarks/new-run
```

单项示例（全局 batch 16）：

```bash
MUSA_VISIBLE_DEVICES=2,3,4,5 tools/musa/run.sh evo1 python tools/musa/benchmark.py \
  --task evo1-stage2 --gpus 4 --batch-size 4 --steps 35 \
  --stage1-checkpoint /workspace/storage/outputs/accept-evo1-stage1-1gpu/checkpoints/000100/pretrained_model \
  --output /workspace/storage/benchmarks/new-run/evo1-stage2-g4-b4
```

`benchmark.py` 支持显式 `--dataset-root`、`--dataset-repo-id`、`--storage`；Pi0.5 的相机 rename map 对应本次公开数据子集，换用其他数据需核实相机名称。每次复测必须使用新的输出路径。

## Pi0.5 每卡 batch 16 / 32 补测

本轮继续使用相同模型、公开毛巾数据子集、BF16、梯度检查点和 GPU AdamW 全量微调（3,616,757,520 参数），无 LoRA、无 CPU offload、无梯度累积；1/2/4 卡分别使用物理 GPU 2、2/3、2/3/4/5。开始时这些卡均无活动计算任务。本轮 6/6 组通过，每组 35 步，剔除前 10 步。

| 卡数 | 每卡 batch | 全局 batch | 端到端秒/步 | 更新秒/步 | 样本/秒 | 峰值 allocated / reserved GiB | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 16 | 3.748 | 2.270 | 4.269 | 32.280 / 33.967 | passed |
| 1 | 32 | 32 | 6.740 | 4.173 | 4.748 | 35.763 / 40.346 | passed |
| 2 | 16 | 32 | 4.775 | 3.025 | 6.701 | 39.291 / 41.469 | passed |
| 2 | 32 | 64 | 7.623 | 5.608 | 8.396 | 42.770 / 47.379 | passed |
| 4 | 16 | 64 | 4.654 | 3.576 | 13.750 | 39.277 / 41.885 | passed |
| 4 | 32 | 128 | 8.796 | 6.496 | 14.552 | 42.770 / 47.379 | passed |

与上一轮每卡 batch 8 的端到端吞吐对照（样本/秒）：

| 卡数 | batch 8（上一轮） | batch 16 | batch 32 |
|---:|---:|---:|---:|
| 1 | 4.43 | 4.27 | 4.75 |
| 2 | 7.19 | 6.70 | 8.40 |
| 4 | 12.08 | 13.75 | 14.55 |

每卡 batch 相同的对照会同时增大全局 batch。与上一轮的差异来自同配置、不同时间的单次短测；服务器未做 NUMA 绑定，数据加载存在波动，细小差值不宜视为确定提升。仍以包含读取和预处理的端到端耗时为主，同时提供更新耗时，避免混用计算吞吐与训练吞吐。

四卡 batch 32 会跨越该小数据集的一轮；继续使用原训练器的 dataloader 和 Accelerate 尾批补齐。新增的 `observed_batch_size` 记录 rank 0 实际动作 batch，验证没有因尾批变小而高估吞吐；该字段从第二个测试起记录，第一个单卡 batch 16 不跨轮。计时边界与 MCCL 计时收集方式未改变。

完整逐步计时、原始日志（仅规范化行尾空白）、参数同步检查、脚本快照见 [`results/benchmark-20261120-pi05-largebatch/`](results/benchmark-20261120-pi05-largebatch/)。服务器结果：`/data/yzw_explore/benchmarks/20261120-pi05-largebatch`。

复现需选择新的输出目录：

```bash
EVORL_BENCH_ROOT=/workspace/storage/benchmarks/pi05-largebatch-new \
EVORL_BENCH_TASKS=pi05 EVORL_BENCH_BATCHES="16 32" \
  bash tools/musa/benchmark_matrix.sh
python tools/musa/summarize_benchmark.py \
  /workspace/storage/benchmarks/pi05-largebatch-new --tasks pi05 --batches 16 32
```

ValueFunction 的单卡/两卡/四卡、每卡 batch 16/32 训练速度另见 [ValueFunction 实测报告](BENCHMARK_VALUE_20261120.md)。
