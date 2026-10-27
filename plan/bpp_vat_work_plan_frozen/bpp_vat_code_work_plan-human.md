# BPP + VAT（BPP 权重不动）施工计划

frozen 表示冻结原 BPP 权重，只训练新增 VAT 参数。policy 和 dataset 使用新类、新文件，原类与原文件保持不变。


修订日期：2026-10-27。当前仅完成计划，尚未修改算法代码。

技术细节见 [AI 版](bpp_vat_code_work_plan-ai.md)。

## 动工计划与实验预期

原 BPP 权重全部保持不变，仅插入并训练 VAT 模块。先做 DrawAnything，再扩展机械臂。

目标是在 BPP 出错的案例上减少回溯、重复和跳段，同时保持正常案例的表现。

VAT 让示教 chunk 的对齐位置单调前进，但不保证注意力峰值或动作单调，仍可能出现死循环。

后续根据结果，再考虑 BPP + VAT 端到端训练。

## 约束

- VAT 只通过 cross-attention 的位置偏置影响动作，不增加 LSTM 特征残差。
- 保留原 Transformer、Diffusion、视觉编码器、池化、位置嵌入及原模型配置。
- 使用连续决策数据，保留跨决策状态；换示教或开始新任务时重置。
- 使用原 diffusion loss，只更新 VAT 参数。
- 首轮采用单 GPU、单环境评估，从轨迹起点训练。

## 源码修改方案

保留整个个人实验源码树，只增加一套 VAT 配置和训练、评估入口。

| 内容 | 处理 |
|---|---|
| 原 BPP 网络 | 保留结构，冻结权重 |
| 原数据读取、环境、视频和评估工具 | 复用，按接入需要小改 |
| LIBERO、UMI、真机及其他模型方案 | 留在仓库，首轮不启用 |
| 数据、权重和已有结果 | 使用现有目录，不复制进代码树 |

首轮不删除源码、不清理依赖、不重写 attention、不开发复杂视频布局。

## 网络接入

```text
当前观测 + 示教特征 + 上次状态
              ↓
     VAT 计算新的对齐位置
              ↓
     给原 cross-attention 加位置偏置
              ↓
     原 BPP 生成动作
```

所有 cross-attention 层共享一组位置偏置参数，减少首轮改动。

## 逐文件施工清单

以下路径相对 `behavior_prompting/train_network/`；实验脚本位于 repo 根目录。

| 文件 | 修改与作用 |
|---|---|
| `model/prompt/vat_alignment.py`（新增） | 实现 VAT 的位置读取、LSTM、位置更新和偏置 |
| `model/prompt/prompt_obs_encoder.py` | 将 VAT 偏置接入原 attention，管理状态和重置 |
| `policy/vat_diffusion_unet_policy.py`（新增） | 新建 `VATDiffusionUnetPolicy`，继承原 policy，负责冻结、VAT 状态和连续训练；原类与文件不改 |
| `common/decision_sequence_sampler.py`（新增） | 按同一执行轨迹连续取样，并固定参考示教 |
| `dataset/vat_draw_image_dataset.py`（新增） | 新建 `VATDrawImageDataset`，继承原 dataset，接入连续采样和验证；原类与文件不改 |
| `workspace/train_policy_workspace.py` | 按连续轨迹训练，保存模型和训练记录 |
| `config/model/obs_encoder/pair_prompt_encoder_vat.yaml`（新增） | 保存 VAT 模块配置 |
| `config/draw_policy_dunetp_vat.yaml`（新增） | 指向新的 policy/dataset 类，保存权重路径、冻结规则和训练设置 |
| `experiments/bpp_vat/`（新增） | 提供 README、train.sh 和 eval.sh |
| `env_runner/draw_runner.py`、远端 `reproduction/core/evaluate.py` | 按需接入新模型并保存对齐位置、指标和停止原因 |

## 训练方式

参考示教 A 和执行轨迹 B 来自同一任务的不同演示。A 在整条轨迹中固定，B 从起点连续输入。

长轨迹分段训练，段间传递状态。沿用原 diffusion loss，原 BPP 权重始终冻结。

冻结网络仍需把梯度传回 VAT，因此不能对 Transformer 和 Diffusion 整体关闭梯度计算。

## 施工顺序

| 阶段 | 工作 | 验收 |
|---|---|---|
| 固定基线 | 核对远端版本、权重和配置 | 保留已有改动，能追溯原版结果 |
| 接入 VAT | 新模块、位置偏置、状态和冻结规则 | 关闭 VAT 时恢复原版行为 |
| 接入连续训练 | 连续采样、分段训练、保存与重载 | VAT 能获得梯度并更新 |
| 小规模验证 | 少量任务、有限训练步数 | 状态正常，原 BPP 权重完全不变 |
| 对照实验 | 原版 BPP vs 同一 BPP + 已训练 VAT | 同案例、同设置比较结果 |

## 验收与留档

- 检查 VAT 参数确实更新，原 BPP 参数和运行统计量保持不变。
- 检查换示教、新任务重置，以及每次决策只更新一次对齐位置。
- 保留正常案例 002，并补充稳定失败、重复图案和较长示教案例。
- 保存原 attention 视频、绘图结果、对齐位置、停止原因、指标和资源开销。
- 每次代码调整记录文件、原因和验证结果；远端修改追加 `agent history/docker修改.md`。
- 训练、推理和仿真均在远端已有 Docker 中运行。

