# Kaggle 私有执行流程

从仓库根目录运行，先 `uv sync --all-extras`。当前账户固定为 `wangshu`。

## 发布源码

```bash
uv run python scripts/kx.py publish-code --message "S0 reader and metrics"
```

首次创建私有 dataset `wangshu/rsna-knee-code`，之后源码有变化时创建新版本；
源码未变时读取远端 manifest 并复用版本。发布后复核私有性和版本。
源码 ZIP 使用固定文件顺序和时间戳，包含 `src/rsna_knee/` 内的 Python 文件
以及 reader 的 LICENSE/NOTICE；`code-manifest.json` 记录每个文件和 ZIP 的 SHA256。
Kaggle 会把 ZIP 展开到同名目录；kernel 加载器兼容 ZIP 原文件和展开后的两种
目录布局，逐项校验源码文件哈希。

Reader 来源 dataset 的许可标识是 Apache-2.0，许可文本和来源声明随它打包。
整个 code dataset 使用 `other` 标识，说明不同文件的权利归各自作者所有。

## 定义、渲染和运行 kernel

每个 kernel 目录包含：

- `kernel.json`：id、title、code_file、GPU/网络设置及其他数据源。
- `kernel.py`：源脚本。
- `config.json`：可选的运行参数。

```bash
uv run python scripts/kx.py render kaggle/s0-smoke --config kaggle/s0-smoke/config.json
uv run python scripts/kx.py push kaggle/s0-smoke --config kaggle/s0-smoke/config.json
uv run python scripts/kx.py wait wangshu/rsna-knee-s0-smoke --timeout 1800
uv run python scripts/kx.py pull wangshu/rsna-knee-s0-smoke experiments/S0-kaggle-smoke/output
```

`render` 只在本地生成 `.rendered/kernel.py` 和 `kernel-metadata.json`。
`push` 验证源码与已发布 ZIP 一致，自动附加 code dataset 和比赛数据，
然后发布私有脚本。渲染脚本在运行前验证源码哈希、加入 `sys.path`，
并通过环境变量 `RK_CONFIG_JSON` 提供配置、`RK_CODE_ROOT` 提供 code dataset 路径。
所有运行配置嵌入单一上传脚本，源脚本可以这样读取：

```python
import json
import os

config = json.loads(os.environ["RK_CONFIG_JSON"])
```

`wait` 默认每 120 秒轮询一次；COMPLETE 时成功退出，ERROR/CANCELLED 时自动
拉取日志并以非零状态退出。`--once` 只查询一次。`pull` 下载全部输出和日志。
Kaggle 的状态/输出 API 当前查询最新运行，所以工具将最新版本与本地回执
或显式的 `owner/slug/version` 对照，不接受静默切换到别的版本。

缓存 kernel 只拉取清单和验收报告，避免下载全部体积文件：

```bash
uv run python scripts/kx.py pull wangshu/rsna-knee-cache-shard-0 experiments/S1/shards/0 \
  --file-pattern '\.(csv|json)$'
```

`--file-pattern` 是文件名正则，日志始终保留。S1 先运行 `cache-probe` 和
`labels`；探针输出存入 `experiments/S1/probe/` 后，运行
`uv run python scripts/s1.py plan` 生成缓存配置和各个 `cache-shard-i` kernel。
`uv run python scripts/run_s1_cache.py --parallel 4 --interval 90` 运行或恢复全量
缓存流程；已有回执和完整归档会复用，待运行的分片优先占用空闲槽位。
全量结束后运行 `uv run python scripts/s1.py verify`。

输出下载仅允许 COMPLETE/ERROR/CANCELLED 等终态，避免运行中误取旧输出。
分页固定为每页 100 项、间隔 3 秒，429/5xx 采用 Retry-After 或指数退避。
中间游标保存在输出目录的 `kx-output-partial.json`，重复运行可续传；成功后删除。

本地 staging 和发布回执在 gitignored 的 `runs/kx/`。实验验收时把相关回执、
输出和指标归档到 `experiments/<id>/`。认证失效时运行 `kaggle auth login` 后重试。

平台字段参考：[dataset 元数据](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md)、
[kernel 元数据](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)。
本地工具使用已安装的 Kaggle Python API，与相应 CLI 子命令调用相同服务。

## 未随仓库发布的生成文件

`teacher/`、`teacher-s6/`、`submit-blend/`、`submit-blend-s6/` 的 `kernel.py` 内嵌了 nartaa 和 heliosli 的公开 notebook，不在此重新分发；用 `scripts/build_teacher_kernel.py`、`build_teacher_s6_kernel.py`、`build_blend_kernel.py`、`build_blend_s6_kernel.py` 从公开 notebook 重新生成。
