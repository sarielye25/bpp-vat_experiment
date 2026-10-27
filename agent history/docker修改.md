# Docker 修改记录

## 执行动作（1）

- 日期：2026-10-27。
- 服务器：`sy.irmv.top:9981`；容器：`wuyixin_bpp`。
- 创建个人项目目录：`/data/yty_vat+bpp_draw`。
- 已确认目录存在；未复制或移动代码、权重、数据，未启动实验。

## 执行动作（2）

- 日期：2026-10-27；容器：`wuyixin_bpp`。
- 将 `/data/BPP/repo` 复制到 `/data/yty_vat+bpp_draw/repo`，保留已有代码改动与依赖；排除 Python 临时缓存。
- 调整副本中 47 个文件的代码、日志路径；原项目未修改。
- 创建 `logs/`、`configs/`、`provenance/` 和 `info.txt`；保存绘图配置、来源 commit、Git 状态及已有修改。
- 数据共用 `/dataset/bpp/draw`；权重共用 `/data/hf/bpp/drawanything_sim`，未重复复制。
- 15 个复现 Shell 脚本语法检查通过；确认评估数据目录及 BPP 权重存在。
- 未安装环境、未运行模型或训练；训练数据完整性尚待检查。

## 执行动作（3）

- 日期：2026-10-27；服务器：`sy.irmv.top:9981`；容器：`wuyixin_bpp`。
- 上传本地 `drawing_prompt.npz` 至 `/data/yty_vat+bpp_draw/inputs/drawing_prompt.npz`；未覆盖已有文件。
- 文件大小 377227 字节；本地与远端 SHA-256 一致：`4802b1d66a32ecdf5e83f514cb4791d740facb0897dbea952a07b2df7127cade`。
- 未运行推理、仿真或训练；未修改共享代码、数据或权重。

## 执行动作（4）

- 日期：2026-10-27；容器：`wuyixin_bpp`。
- 检查个人副本导入路径与子模块，通过；使用现有 base 环境，限制 CPU 数学库线程为 2。
- 启动单次原版 BPP 绘图测试：输入 `inputs/drawing_prompt.npz`，attention v2 开启、画板角度 0°、最多 400 步，保留转换后的 Zarr。
- 输出：`/data/yty_vat+bpp_draw/logs/draw_origin/20261027_115930`（容器时间 UTC）；启动日志为同级 `20261027_115930_launcher.log`。
- 启动 PID：525579。运行结果待验证；未训练、未修改共享代码或容器配置。
- 结果：测试完成，结果验证脚本通过；7 次预测，84 帧 rollout，attention 视频 1616×1088、10 fps、8.4 秒，ffprobe 确认 H.264 / yuv420p。
- 单示教最大负对称 Chamfer 分数：-17.40328388104558；不代表 benchmark 或成功率。
- 结果已下载到本地 `results/draw_origin/20261027_115930/`，包含视频、最终图片、配置、指标及日志。
- 注意：原始示教 87 步，现有 prompt 视频为 80 帧、attention 网格 8 个片段，末尾分块处理需要在正式对比前核对。

## 执行动作（5）

- 日期：2026-10-27；容器：`wuyixin_bpp`。
- 上传 `draw prompt/draw_prompt_002.npz` 至个人 `inputs/draw_prompt_002.npz`，376 步 / 37.6 秒；SHA-256 双端一致：`4557478a1bfe5e83c8f4d1f3f17ddb4208cd18952d6c96c6400fac34376c4f6f`。
- 启动原版 BPP 单示教测试，attention v2、画板 0°、最多 400 步、保留 Zarr，线程限制 2；PID 546515。
- 远端输出：`/data/yty_vat+bpp_draw/logs/draw_origin/draw_prompt_002/20261027_202053_HKT`，时间明确使用香港时区。
- 本地输出计划：`result/draw_prompt_002/20261027_202053_HKT/`。未修改共享代码或训练。
- 结果：测试完成并通过现有验证，204 帧 / 20.4 秒；attention 1616×2432，ffprobe 确认 H.264。
- 单案例最大负对称 Chamfer 分数：-1.572211487224516；不代表 benchmark。
- 视频、图片、配置和日志已下载到计划的本地 result 目录，逐文件 SHA-256 验证通过；新增便于打开的 `attention.mp4`、`rollout.mp4`、`prompt.mp4` 副本与 README.txt。
