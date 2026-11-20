# S5000 MUSA 适配实测报告

2026-11-20，`worker31096`，容器 `yzw_explore`。按最新要求，验收范围为**单卡、双卡**。
所有模型级测试使用真实公开预训练权重和真实毛巾任务数据，没有用随机骨干替代预训练验收。

## 结果

| 路径 | 单卡 | 双卡 | 结果 |
|---|---|---|---|
| Pi0.5 | 100 步，恢复至 105 步 | 20 步 | 通过 |
| Evo1 Stage1 | 100 步，恢复至 105 步 | 20 步，恢复至 25 步 | 通过 |
| Evo1 Stage2 | 100 步，恢复至 105 步 | 20 步，恢复至 25 步 | 通过 |
| PiStar06 ValueFunction | 100 步，恢复至 105 步 | 20 步 | 通过 |

每卡 batch=1，原生 MUSA + Accelerate/DDP + MCCL，BF16，关闭 FlashAttention 和 torch.compile。
检查了 loss 有限、梯度、真实 checkpoint 的优化器 step，以及中间和最终 checkpoint 的权重变化。
双卡真实模型的参数抽样跨 rank 完全一致；MCCL all-reduce/all-gather/barrier 和完整小型 DDP 参数一致性也通过。
显存按 rank 记录于原始 JSON；Pi0.5 单卡约 32.1 GiB、双卡每卡约 39.1 GiB。
这些是功能验收，不是收敛或吞吐缩放基准：单卡与双卡的全局 batch 不同。

- 相关回归：**103 passed，19 skipped**。跳过项为不可用 CUDA/MPS 或缺失的本地测试数据。
- MUSA BF16 矩阵相对 CPU FP32 检查通过，最大绝对误差 `0.0883541107`；Pi0.5 时间编码相对 CPU 误差为 0。
- Pi0.5 原生 Beta 采样、带 mask 注意力反向、MUSA RNG safetensors 保存/恢复通过，没有引入全局 CPU 算子回退。
- 修改文件的 Ruff 检查、Python 编译检查通过。
- 三套任务依赖闭包审计均通过；基础镜像中无关 serving 包的全量 `pip check` 冲突单独保留，未冒充全量清洁环境。

## 标注和迭代

源数据固定到 `Elvinky/bi_so101_fold_towel_d0@71edae2283713bbf380cab4a6a2f7fe8bba7d13d`。
抽取原始 episode 0、1，保留 3 路视频和原始 `episode_success=success` 标签，共 **3432 帧**。
真实失败轨迹不在这个小子集中；失败标签、奖励和边界逻辑由单元测试覆盖。

单/双卡标注对比固定使用价值模型**第 100 步** checkpoint，而不是会随恢复训练移动的 `last`：

| 检查 | 结果 |
|---|---|
| 帧覆盖 | 3432 帧，索引唯一且完全一致 |
| value 最大绝对误差 | 0 |
| advantage 最大绝对误差 | 0 |
| indicator 一致率 | 100% |

Pi0.5 和 Evo1 分别完成：复用已验收的 SFT/value checkpoint → 读取真实价值标注 →
20 步 ACP 更新 → 重新加载 → 动作推理 → 导出部署配置。
训练日志确认 `Advantage: negative` 标签经过文本输入；正/负标签以及 Pi0.5 tokenizer 路径另有回归测试。
输出动作均为 `[1, 12]`，数值有限。

下一轮将标注后的原始小子集合法合并两次，共 **4 个 episode、6864 帧**，检查重编号和真实标签。
两类策略均从上一轮 ACP checkpoint 再训练 5 步，并通过动作推理和部署配置导出。
它们明确标记 `replayed_data=true`，是流程重放测试，不代表新采集的机器人经验。
整个验收没有操作真实机器人，也不推断真机成功率提升。

## 关键适配

- MUSA 设备注册和索引校验；显式 MUSA 请求失败时报错；模型和 processor 服从实际 rank 设备。
- Evo1 BF16/autocast；价值模型按目标设备选择 dtype；CUDA/cuDNN 优化配置与 MUSA 隔离。
- safetensors 显式 CPU 反序列化后复制到模型设备，解决其不识别 `musa` 名称的问题。
- 保存每个 MUSA rank 的 RNG；支持完整优化器/调度器恢复。数据迭代器仍沿用原项目重启行为，不宣称样本顺序逐位续接。
- InternVL3 视觉 checkpoint 实例局部改为非重入实现，避免修改全局 torch 或缓存的远程源码。
- 冻结 SigLIP 未使用的文本/对比参数，修正价值恢复时不存在的 action unnormalizer 覆盖。
- Pi0.5 固定 OpenPI Transformers 提交、原始公开 PaliGemma tokenizer、摄像头映射，以及官方 checkpoint 中禁用的动作转换步骤。
  `enabled=true` 的相对/绝对动作转换会明确拒绝，不静默改变动作语义。
- 远程推理接口注册 Evo1，在构造模型前设置设备、禁用梯度、返回 CPU 动作。服务和采集 CLI/接口测试通过；未启动硬件客户端。
- 同步脚本排除生成的 egg-info/dist-info；安装器显式刷新源元数据，避免旧元数据覆盖新依赖约束。

## 环境与位置

Python 3.10；PyTorch/torch_musa 2.9.0，vendor torchvision 0.22.1；MUSA toolkit **4.3.5**；
宿主驱动 **5.2.0-server**；实际加载 MCCL **2.11.4**，不是 stub。NumPy 固定 1.26.4 以匹配供应商 torch ABI。
固定基础镜像：

```text
registry.mthreads.com/devtech/glm5.2@sha256:8b50b9e2020f68488e380e7ee3484af6cc1ab28e695ce306240ff7cb719eff3e
```

| 内容 | 宿主路径 | 容器路径 |
|---|---|---|
| 源码 | `/home/mccuseradmin/yzw_explore/Evo-RL` | `/workspace/Evo-RL` |
| 环境、模型、数据、日志 | `/data/yzw_explore` | `/workspace/storage` |

环境分别为 `environments/pi05`、`environments/evo1`、`environments/value`。
权重、数据和输出不写入 `/home`，也不提交到 Git。没有修改其他容器或宿主驱动。
原始训练日志、checkpoint 和两轮输出保留在 `/workspace/storage/logs`、`outputs`。

可审阅的精简证据在 [`results/2026-11-20`](results/2026-11-20)，包括：
模型 step/优化器/权重变化审计、逐 rank 显存、标注对比、四轮动作/部署配置、依赖审计、公开资产 revision 和回归结果。
环境锁在 [`locks`](locks)，运行说明在 [`README.md`](README.md)。

MUSA 首版采用 PyAV、无可视化的训练路径；TorchCodec/Rerun 是可选 extra，未在这套 NumPy 1.26 的服务器环境中验收。
FlashAttention、torch.compile、FSDP、长时间稳定性及真机闭环不属于本次单/双卡功能验收结论。
