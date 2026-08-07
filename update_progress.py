#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成汇总进度报告 PROGRESS.md（读全部 shard 的 state 文件）。"""
import json
import os
import sys
import time

if sys.stdout.encoding.lower() in ("gbk", "gb2312", "cp936"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = os.environ.get("HF_OUTPUT_DIR", "modelscope_output")
SHARDS = int(os.environ.get("SHARDS", "5"))

lines = [
    "# HuggingFace 组织/用户画像采集进度",
    "",
    f"- **更新时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
    f"- **分片数**: {SHARDS}",
    "",
]
done_total = org_total = user_total = err_total = all_total = 0
rows = []
for i in range(SHARDS):
    f = os.path.join(OUT_DIR, f"state_hf_profiles_{i}.json")
    try:
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        done_total += s.get("count", 0)
        org_total += s.get("orgs", 0)
        user_total += s.get("users", 0)
        err_total += s.get("errors", 0)
        all_total += s.get("total", 0)
        rows.append(
            f"- 分片 {i}: {s.get('count', 0):,} / {s.get('total', 0):,}，"
            f"orgs={s.get('orgs', 0):,}，users={s.get('users', 0):,}，err={s.get('errors', 0):,}"
        )
    except Exception:
        rows.append(f"- 分片 {i}: 尚未开始")

pct = done_total / all_total * 100 if all_total else 0
lines += [
    f"- **总体进度**: {done_total:,} / {all_total:,} 个 owner（{pct:.1f}%）",
    f"- **组织数**: {org_total:,}",
    f"- **个人用户数**: {user_total:,}",
    f"- **失败数**: {err_total:,}",
    "",
    "## 各分片进度",
    "",
]
lines += rows
lines += [
    "",
    "> 本报告由爬虫每 30 分钟自动生成并提交。",
]

with open(os.path.join(OUT_DIR, "PROGRESS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"PROGRESS.md: {done_total}/{all_total} ({pct:.1f}%)")
