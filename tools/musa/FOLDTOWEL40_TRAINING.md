# FoldTowel 40 轨迹正式训练

批准的配方使用 40 条真实轨迹、47,398 帧，所有轨迹结果标记为真实成功。数据来自 4090 的 `/home/MINT-Minhang/work/YeZewei/RWorld-RL/dataset/FoldTowel_1119`；复制 `meta/`、`data/`、`videos/`，逐文件 SHA-256 验证后放置在容器 `/workspace/storage/data/FoldTowel_1119`。原始数据和其他项目保持完整。

| 参数 | Pi0.5 | ValueFunction |
|---|---:|---:|
| 物理 GPU | 2,3,4,5 | 6,7 |
| 每卡 batch / 全局 batch | 32 / 128 | 16 / 32 |
| Epochs / steps | 25 / 9,275 | 10 / 14,820 |
| Peak LR / floor LR | 2.5e-5 / 2.5e-6 | 5e-5 / 1e-6 |
| Warmup / decay steps | 500 / 9,275 | 500 / 14,820 |
| 每 epoch 保存间隔 | 371 | 1,482 |
| 分布式端口 | 29621 | 29622 |

两项均原生 MUSA、Accelerate/DDP/MCCL、BF16、GPU AdamW、梯度检查点、梯度累积1、PyAV、`num_workers=0`、无图像增强，日志间隔10（用于及时确认真实训练与在线指标）。Pi0.5 全量训练，SigLIP + Qwen2.5-0.5B ValueFunction 的视觉和语言骨干均解冻。ValueFunction 使用 `legacy` 真实回合结果监督。Pi0.5 第三相机为本数据集的 `right_top`，显式映射至 `base_0_rgb`。

SwanLab 项目复用已核实的 `sixpigs/RWorld-RL`。SDK 固定0.10.6，新增依赖见 `locks/tracking-extra.txt`；安装前对两套环境审查 pip 计划，禁止更换 torch、torch_musa、torchvision、NumPy、Transformers 或 Accelerate。`validate_environment.py` 会检查已安装 SwanLab 的依赖闭包。

## 启动与安全处理

`recipes/foldtowel40/{pi05,value}.json` 保存完整参数，无凭据。运行目录为 `/workspace/storage/runs/foldtowel40-20261120/{pi05,value}`，各有 `plan.json`；`check_training_plan.py` 用原生 CLI 解析它们，验证 epochs/steps、学习率调度及设备设置，且不启动训练。

数据目录必须有已完成文件哈希校验的 `transfer-validation.json`，启动器拒绝测速小子集和未验证的数据。

在容器中：

```bash
cd /workspace/Evo-RL
python tools/musa/training_job.py --start /workspace/storage/runs/foldtowel40-20261120
```

脚本通过隐藏的交互输入（或 stdin）接收 API key，只把它放入后台进程环境。原生 SwanLab logger 使用 `login(save=False)`；新 SDK 使用 online 模式，初始化在线失败时抛错，显式选择 SwanLab 时不会静默切换到 W&B。训练 stdout/stderr 在写入管理日志前对 key 做替换过滤。凭据不写入参数、仓库、启动脚本或命令行参数。

启动器检查 GPU 组和分布式端口互不重叠，按卡加锁，检查空闲计算状态后启动。两个 supervisor 通过独立进程会话留在容器中，SSH 断开后继续运行。运行时保留实测使用的容器 UID 0，进程 GID 设为宿主 `users` 组100，umask 为002，便于宿主账号管理新输出；不修改宿主设备权限。它们保留 `train.log`、`supervisor.log`、`state.json`；`launch.json` 记录 supervisor PID。重复启动会被拒绝，需先检查既有状态。

## 检查状态及输出

```bash
cat /workspace/storage/runs/foldtowel40-20261120/pi05/state.json
cat /workspace/storage/runs/foldtowel40-20261120/value/state.json
tail -n 10 /workspace/storage/runs/foldtowel40-20261120/pi05/train.log
tail -n 10 /workspace/storage/runs/foldtowel40-20261120/value/train.log
```

输出分别位于：

- `/workspace/storage/outputs/foldtowel40-pi05-4gpu-20261120`
- `/workspace/storage/outputs/foldtowel40-value-2gpu-20261120`

原生训练器每 epoch 和最终步保存完整 checkpoint，包含模型、优化器、调度器及各 rank RNG；旧检查点保留。SwanLab 同步训练配置、loss、梯度、学习率和吞吐相关计时等标量，模型 checkpoint 保存在服务器。

恢复训练时使用完整 checkpoint 的 `train_config.json` 或 `value_train_config.json`，设置 `--config_path=... --resume=true` 并保留同一卡数和原 SwanLab run ID。先核实 checkpoint 和 GPU 状态，再启动恢复进程；不重复执行 fresh-start 启动器。
