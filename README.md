# HuggingFace 云端爬取仓库

在 GitHub Actions 上运行 HuggingFace 数据采集，产出以 artifact 形式下载。

## Workflows

| 文件 | 内容 | 触发方式 |
|---|---|---|
| `01_crawl_hf_full.yml` | HF 全量清单（models/datasets/spaces），断点续爬 | workflow_dispatch |
| `02_update_match.yml` | 更新跨平台匹配表（依赖 01 的产物） | workflow_dispatch |
| `03_crawl_hf_subset.yml` | HF 子集行为数据（commits/discussions/README/tree） | workflow_dispatch |
| `04_crawl_hf_governance.yml` | HF 治理文档 | workflow_dispatch |
| `05_crawl_hf_profiles.yml` | HF 组织/用户画像（依赖 01 的模型清单） | workflow_dispatch |
| `hf_counterpart.yml` | HF ↔ 魔搭对照采集（每日定时） | workflow_dispatch + schedule |

## 使用

1. 进入 GitHub 仓库 Actions 页面
2. 选择对应 workflow → Run workflow → 手动触发
3. 运行完成后在 workflow run 页面下载 artifact

## 输出约定

云端统一输出到 `modelscope_output/`（CI 工作区，不与本地魔搭数据混放），下载后按本地约定归档到 `02-原始数据/01-HuggingFace数据/data/huggingface/`。
