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
