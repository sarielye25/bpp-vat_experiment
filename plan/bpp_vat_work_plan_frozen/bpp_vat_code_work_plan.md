# BPP + VAT （bpp权重不动）动工计划

frozen 表示冻结原 BPP 权重，只训练新增 VAT 参数。policy 和 dataset 使用新类、新文件，原类与原文件保持不变。


修订日期：2026-10-27。

## 动工计划&实验预期
原 BPP 权重全部保持不变，仅插入并训练新增 VAT 模块。
后续可能会进行BPP+VAT端到端训练，调整bpp权重参数。
目标是在 BPP 出错案例上减少回溯、重复和跳段。VAT 让 chunk 对齐位置单调前进，但不保证注意力峰值或动作单调，仍可能出现死循环。

## 约束

1. VAT 的影响需要通过 cross-attention 传到动作输出；不再增加 LSTM 特征残差注入。
2. LSTM 需要连续决策数据和跨决策状态。冻结 BPP 不能省去连续采样、状态 reset 和训练递推。
3. 使用原 diffusion loss 训练 VAT，需要梯度穿过冻结 Transformer 和 Diffusion；它们的权重不更新，但其前向不能整体放进 `no_grad()`。
4. BPP 原 attention 已接受 `memory_mask`，首版复用加性 mask 接口，不重写 attention 算子。

## 源码修改方案

保留整个个人实验源码树。日常只用一套 VAT 配置和 train/eval 脚本，不为删除无关文件引入额外重构。

| 内容 | 首轮处理与原因 |
|---|---|
| Pair tokenizer/Prompt Encoder、原视觉编码器、原 Transformer、Diffusion U-Net | 保留结构和权重；仅在 Prompt Encoder 接入位置偏置 |
| ReplayBuffer、原 sampler/Draw dataset、normalizer、增强与张量工具 | 保留；连续采样复用字段和轨迹索引，原版数据路径仍可回归 |
| train/workspace、Draw env/runner、gym/video、prompt/attention 工具 | 保留已有运行与结果验证链 |
| 原配置及 Hydra defaults、旧类名兼容代码 | 保留，避免旧 checkpoint 加载失败 |
| 远端 reproduction 的转换、启动、evaluate、verify 工具 | 复用已验证入口；仅在确有接入需要时小改 |
| LIBERO/LIBERO-Gen、UMI、真机、ICRT、动作 Transformer、语言/目标图像 baseline | 留在仓库，首轮不实例化、不训练、不评估 |
| LICENSE、setup、依赖/子模块与来源文档 | 保留，便于追溯和第二阶段扩展 |
| pycache、临时产物、旧日志、大型数据与权重 | 不纳入代码同步清单；现有数据/权重仍从已有路径引用 |

明确暂不做：物理删目录、精简安装环境、重写公共 workspace、分拆视觉编码器、为 Draw-only 重构所有任务导入。

已知依赖关系留档：`utils/load_env.py` 顶层导入 LIBERO，`plot_util.py` 导入动作 Transformer policy，视觉编码器导入 CLIP 文本类。因此“首轮不用”不等于“可以删除”。已跑通环境先保留；只有实际阻塞新入口时才做最小延迟导入修复，并单独记录。

## 逐文件施工清单

除 `experiments/` 与 `reproduction/` 外，下表路径相对 `behavior_prompting/train_network/`。新增文件均为计划，尚未实施。

| 文件 | 修改及作用 |
|---|---|
| `model/prompt/vat_alignment.py`（新增） | 集中实现位置读取、LSTM/步长、共享 IRPB、显式状态与诊断；不做残差注入 |
| `model/prompt/prompt_obs_encoder.py` | 默认关闭的 VAT 开关；构造并传递 `memory_mask`；接状态、缓存/reset、无状态 shape 查询；原参数名称与输出尺寸保持不变 |
| `policy/vat_diffusion_unet_policy.py`（新增） | 新建 `VATDiffusionUnetPolicy(DiffusionUnetPolicy)`，承接序列损失、VAT 状态、冻结与训练模式、VAT-only optimizer；复用原去噪网络与损失公式，原 policy 类与文件不改 |
| `common/decision_sequence_sampler.py`（新增） | 复用 ReplayBuffer 索引/字段定义生成 A/B 配对与从起点开始的连续决策，不跨 episode |
| `dataset/vat_draw_image_dataset.py`（新增） | 新建 `VATDrawImageDataset(DrawImageDataset)`，承接连续采样、序列取样和验证集构造；复用原数据转换/normalizer，原 dataset 类与文件不改 |
| `workspace/train_policy_workspace.py` | 最小 VAT 分支：绕开随机按任务组批、TBPTT backward、VAT-only optimizer 与 checkpoint、关闭 EMA/随机 action-MSE；保留原 BPP 训练路径 |
| `config/model/obs_encoder/pair_prompt_encoder_vat.yaml`（新增） | 原 encoder 配置 + 新增 VAT 参数，原 Transformer 超参数不变 |
| `config/draw_policy_dunetp_vat.yaml`（新增） | 用 `_target_` 选择 VAT policy/dataset；保留冻结规则、权重来源、序列训练、batch=1/单 GPU/FP32/无 EMA、单环境评估；原 YAML 不改 |
| `experiments/bpp_vat/{README.md,train.sh,eval.sh}`（新增） | 简单注释入口与配置说明；记录个人路径、版本、冻结验证、VAT 轨迹和结果 |
| `env_runner/draw_runner.py`（仅按需小改） | 保存 p/delta、决策次数和停止原因；仍用原 attention 视频；首轮不开发复杂叠加可视化 |
| 远端 `reproduction/core/evaluate.py`（仅按需） | 复用旧类名/长度兼容逻辑，加载新 VAT 配置与参数；不重写原版启动链 |

首轮**不改**：原 `policy/diffusion_unet_policy.py` 与 `dataset/draw_image_dataset.py`（含原类定义）、`transformer_decoder_with_attention.py`、Diffusion U-Net 模块、视觉网络结构、`utils/load_env.py`、`plot_util.py`/视频布局、LIBERO 文件、公共 checkpoint 框架；不新增通用 checkpoint 兼容层或序列 collate。实际接入出现阻塞时，先说明具体原因，再做局部调整并留档。

## 最小网络接入

```text
示教 A -> 冻结 tokenizer/视觉编码/池化 -> H 与 padding mask
当前观测窗口 -> 冻结视觉编码 -> X_t
旧位置 p、旧 LSTM 状态 + H、X_t
    -> location-only 位置读取 -> LSTM -> softplus 步长 -> 新位置 p_t
    -> 共享 IRPB(p_t - j)
    -> 原 cross-attention 分数 + 位置偏置
    -> 冻结 Prompt Transformer -> z_t
    -> 冻结 Diffusion U-Net -> 原 diffusion loss
                                    |
                         仅更新新增 VAT 参数
```

### 保持不变与允许训练的部分

| 部分 | 参数与运行约定 |
|---|---|
| 原 BPP 所有参数 | `requires_grad=False`；不进入 optimizer；训练前后逐 tensor 比较 |
| 原模型 buffers/normalizer | 保持不变，避免 BatchNorm 等在 train 模式更新统计 |
| 原视觉/示教特征提取 | 固定 eval 模式；无可训练输入路径时可用 `no_grad()` 生成普通 detached 特征 |
| 原 Prompt Transformer、Diffusion | 固定 eval 模式，保留对输入的 autograd；eval 不会阻止梯度回传 |
| 新增 VAT | 训练 LSTM、步长投影、必要特征投影、alignment IRPB 与内容 attention IRPB；optimizer 仅包含这些参数 |

不能对整个 policy 直接调用 `eval()` 后就忘记恢复 VAT 的训练模式，也不能调用全模型 `train()` 后让冻结 BPP 的 dropout/buffers 改变。每次模式切换显式保持 BPP eval、VAT train/eval。

### 位置更新与共享偏置

- 沿用 `merge_prompt_tokens=obs_and_action`，一个示教片段对应一个位置 j；不改变原 pooling 设置。首版只支持这条路径。
- `p_init=0`，LSTM h/c 初始为零。先按旧位置读取 H，再结合当前观测计算 `p_t=p_prev+softplus(delta_logit)`；同次决策的全部观测 token 共用 p_t。
- alignment read 的 logits 只有位置 IRPB，无 QK 内容比较；对有效示教特征加权读取。必要投影属于新增 VAT。当前观测用固定均值池化形成 x，再按需要投影。
- **不加 LSTM 残差特征、不改原 Q/K/V、不改原 self-attention。** VAT 只通过位置偏置影响 BPP。
- 内容 cross-attention 首版共享一张每头 IRPB 表：所有层共用同一组偏置及同一 p_t；alignment read 使用独立 IRPB 表。共享跨层表是减少参数/接线的实验简化，与论文逐层独立表不同，报告时明确标注。
- 一次 `predict_action()` 只推进一次 VAT；所有 diffusion 去噪步共享同一个 z_t。原 action/obs horizon、执行跨度与去噪设置不改。

### IRPB、mask 与初始化

- 按论文式 (3)(4)保留近距线性、远距对数分桶与邻桶插值；按式 (7)加入最大距离惩罚。B、D_max、sigma、MDP 强度只放新增 VAT 配置。
- 配置明确 `buckets_per_side=B`、表长 `2B-1`；有符号桶用显式索引偏移。末桶截断、零点和 `log(0)` 需妥善处理。
- Gaussian 初始化是桶索引空间的 log bias：`-0.5*(bucket/sigma)^2`。这些初始偏置会立即改变 attention；未训练的 VAT 不应被当作已经能改善 BPP。
- 在 Prompt Encoder 中生成 `[B*n_heads, query_tokens, memory_tokens]` 浮点 `memory_mask`，将位置偏置与 padding 的 `-inf` 合并；避免再传 dtype 不一致的重复 mask。原 decoder 已逐层传递 `memory_mask`，首版不修改 decoder 文件。
- 首轮 Draw 沿用 checkpoint 的 sink 设置。通用规则保留：sink 不参与 alignment read、不占片段位置，内容 attention 的 sink 偏置为零；全 padding 输入报错。
- p/h/c、距离分桶优先 FP32；attention mask 按调用 dtype 转换并保留梯度。位置不硬夹到最后片段，记录越界；不额外引入位置监督或终止规则。
- VAT 关闭或测试时强制 bias 为零，必须恢复原 attention 路径；这与正常 Gaussian 初始化的行为分开验证。

## 最小状态与连续训练链

### 状态

新增 VAT 提供显式接口：`step(X, H, mask, state) -> (bias, next_state, diagnostics)`，state 为每条执行序列的 p/h/c。

- 训练每条轨迹从零初始化；段内传递 state，TBPTT 段间 detach；不同随机 batch 不共享状态。
- 推理 state 可缓存，`reset()` 和更换 `prompt()` 清除旧状态；`prompt()` 编码示教不推进位置。
- `output_shape()`/样例 forward 不得推进持久状态，优先无状态尺寸推导或隔离临时 state。
- 首轮 eval 只开一个环境，不实现向量槽部分 reset、自动重置或 canvas restore；关闭这些额外路径。以后扩展多环境再做逐槽 state/mask。
- 冻结 BPP 的特征可在单条训练轨迹内复用，缓存点须在 VAT 之前。训练使用普通 detached tensor；不要直接复用 `inference_mode()` 产生、无法被 autograd 保存的缓存。首版不增加离线特征库。

### 数据和 loss

- 新增连续决策采样路径，保留旧 sampler。每条样本固定示教 A 与执行 B，来自同任务、同训练 split 的不同 episode；没有合格配对时报错。
- 按实际执行 horizon 取连续决策，Draw 默认间隔 12 个控制步；观测窗口 2、预测动作 16、示教 chunk 10 是不同维度，实际均以 checkpoint 配置为准。
- 每个样本覆盖从起点到结束的执行轨迹；以 K 个决策分段 TBPTT 控制显存，不只训练固定短前缀。首版可以 batch size=1，避免为了批量 padding 增加复杂度。
- 每段串行计算 VAT，复用原 diffusion 噪声采样/target/MSE。按整条轨迹有效决策数归一化，各段 backward 后 detach state，轨迹结束才 `optimizer.step()`；不在状态链中间换参数。
- 训练只优化 VAT；首版关闭 EMA、分布式训练、混合精度与原随机 action-MSE 评估入口，先用 FP32 小规模验证，避免冻结权重被 EMA 或模式切换意外改动。需要加速时另记变更。
- 验证也从起点递推；不在随机中段把状态重置为零。随机窗口预热、变长 batch padding、多 GPU 均延后。
- `train_diffusion_n_samples` 若原配置启用，只在 z_t 生成后复制条件，不能重复推进 VAT。原损失目标不变，不加位置标签或对齐辅助 loss。
- 首轮沿用冻结 BPP 的 eval 特征预处理，保持原归一化/几何定义；不新增随机几何增强。后续使用增强须保证序列内一致。

原 `DiffusionUnetPolicy.compute_loss()` 按独立时刻工作且拒绝 `valid_mask`，保持该方法不变；在新 `VATDiffusionUnetPolicy` 中覆盖序列训练方法，组织状态并复用原单决策损失逻辑。首版 batch=1 无决策 padding；以后批量化才加入独立 decision mask，不能与 prompt mask 混用。

## 权重加载与“原权重不动”的验收

1. 复用现有兼容代码加载明确选定的 BPP checkpoint 分支（model 或 EMA），与原版评估一致。保留来源文件、SHA-256、配置和迁移报告。
2. 加载时仅允许白名单中的新增 VAT 参数缺失；原 BPP missing/unexpected keys、shape mismatch 均失败。原 normalizer、位置表、pooling、n_emb、horizon 全部对齐。
3. 先冻结原 BPP 再仅开放 VAT；检查所有 optimizer 参数恰好属于 VAT、无重复和遗漏。清除原参数梯度，验证训练后仍为 None。
4. loss 必须传到 VAT；冻结 Transformer/Diffusion 不使用整体 `no_grad()`，其输入梯度应存在。冻结特征无需保留梯度；不能要求原视觉特征仍连着计算图。
5. 训练前保存加载后的原 BPP 参数和 buffers 基准值，训练/保存重载后逐 key 用精确 tensor 比较确认全部不变。不能用整个新 checkpoint 文件 hash 判断，因为新增 VAT 和训练元信息必然变化。
6. 新实验 checkpoint 保存完整模型参数及配置、VAT optimizer/scheduler/step/随机状态；原参数仅作相同值的快照，不覆盖原始权重文件。复用现有保存框架；最小恢复需验证重载后的输出一致，段中断点精确续训延后。
7. VAT 关闭、固定输入与随机种子时原输出应匹配。保留原绝对位置表，因此首版不宣称支持超出表长的任意长示教。
8. Draw 原 chunk 截尾规则保持一致（87→80、376→370）；不同时给 VAT 改尾块 padding。更改数据规则须独立评估。

冻结会减少 optimizer 状态和权重梯度开销，但梯度仍需穿过原网络，不能据可训练参数比例推断显存/耗时降幅。

## 实施顺序与最小验收

| 阶段 | 做什么 | 完成标准 |
|---|---|---|
| P0 固定基线 | 读远端 info/AGENTS，核对本地/远端版本、个人导入路径和 checkpoint 配置 | 不覆盖已有改动；基线、来源及配置可追溯 |
| P1 最小接入 | VAT 模块、共享 mask、状态与冻结规则 | 关闭 VAT/零 bias 回归通过；原权重与 buffers 不变 |
| P2 连续训练链 | 单 GPU、batch=1、起点递推与 TBPTT、VAT-only 优化 | 时间索引/配对正确；分段状态一致；VAT 梯度有效，原参数无梯度 |
| P3 小规模验证 | 获训练授权后有限步 smoke/拟合检查与保存重载 | loss/梯度有限，状态/reset 正确；训练前后原权重精确一致 |
| P4 同条件对照 | 冻结原版 BPP（VAT 关闭）vs 同一 BPP + 已训练 VAT | 固定案例/seed/画板/停止/chunk；报告任务表现、进度与成本，不要求微调 BPP |
| P5 后续扩展 | 根据结果决定独立分层 IRPB、多环境、随机窗口、机械臂 | 各扩展单独记录与验收，不作为首轮必要工作 |

必要测试：IRPB 正负/零点/桶边界/末桶/MDP 和非折点梯度；padding/sink mask；重复去噪不推进状态；reset/换示教/shape 查询不泄漏状态；序列不跨 episode；TBPTT 与逐步前向数值一致；VAT 参数确实更新且原 BPP 参数/buffers 精确不变。

诊断先保存为 JSON/NPZ/CSV，无须先开发视频叠加：p_before/p_after/delta、有效示教长度、attention argmax/期望位置、越界率、停止原因、Chamfer、耗时和峰值显存；保留原 attention/rollout H.264 视频。

002 用于正常表现保真；001 是否属于对齐失败仍需确认。补充稳定失败和重复图案案例；较长示教仍在原位置表范围内。未训练 VAT 仅用于接线检查，不据其表现得出方法优劣结论。

所有模型训练/推理/仿真与涉及模型运行的验证在远端既有 Docker 完成；本地负责编辑与静态检查。当前任务仅改计划，不启动实验。
## 新类与原类

| 原类（保持不变） | 新类 | 新文件 |
|---|---|---|
| `DiffusionUnetPolicy` | `VATDiffusionUnetPolicy` | `policy/vat_diffusion_unet_policy.py` |
| `DrawImageDataset` | `VATDrawImageDataset` | `dataset/vat_draw_image_dataset.py` |

优先继承原类，复用已有功能，只覆盖 VAT 需要改变的方法。原版 BPP 配置继续使用原类，新 VAT 配置选择新类。
### 配置选择

以下为覆盖字段；其余配置沿用已核对的 checkpoint/Hydra 配置。

```yaml
model:
  _target_: behavior_prompting.train_network.policy.vat_diffusion_unet_policy.VATDiffusionUnetPolicy
task:
  dataset:
    _target_: behavior_prompting.train_network.dataset.vat_draw_image_dataset.VATDrawImageDataset
```

### 继承与兼容要求

- `VATDiffusionUnetPolicy` 复用原初始化、归一化、去噪与动作输出；仅覆盖序列训练、必要的状态/reset、冻结与 train/eval 模式、optimizer 构造。现有单步方法可直接满足新路径时继承使用，避免复制整份 policy。
- `VATDrawImageDataset` 复用 ReplayBuffer、字段处理及 normalizer；在子类内接入新 sampler，按需覆盖初始化、`__len__`、`__getitem__`、`get_validation_dataset()` 及相关索引接口。原父类验证方法会重建原 `SequenceSampler`，不能不加检查地继承；训练/验证都必须保留新的序列语义和数据划分。
- 若调用父类初始化会先构造旧 sampler，先核对成本；首轮允许轻量初始化后在子类替换 sampler。确有大量重复加载时再局部解决，保持两个原文件不变。
- 保留原 `model`、`obs_encoder` 等属性名称和 state_dict key；新增 Python 类不要求新增一层模型包装前缀。仅新增 VAT 参数允许缺失，旧 BPP 参数必须全部正确加载。
- 训练入口必须从新 YAML 实例化新类，再迁移旧 checkpoint 权重。只读取旧 checkpoint 中的原 `_target_` 会构造原类，不能据此运行 VAT；旧类名兼容逻辑继续保留。
- Prompt Encoder 的 VAT 偏置接入和 workspace 的必要训练组织仍按现有清单实施；“原类不改”本次明确针对 policy/dataset 两个文件，不额外扩成全仓库重构。

### 新增验收

- Git diff 确认原 `policy/diffusion_unet_policy.py`、`dataset/draw_image_dataset.py` 无改动。
- 原 YAML 实例化原类；VAT YAML 实例化上述新类。新 dataset 的验证集同样使用序列 sampler。
- 原配置、原 checkpoint 和原版推理回归通过；新类加载后的原 BPP state_dict key/值一致，训练后参数与 buffers 仍精确不变。
## 本次计划调整

- 按学长建议，将 policy/dataset 的 VAT 改动放入两个新增子类和文件，保留原类、原文件及原版配置。
- frozen 统一指冻结 BPP 权重；本文件夹保存该方案。
- 本次仅更新文件夹内三份计划，未实施代码或远端修改。
