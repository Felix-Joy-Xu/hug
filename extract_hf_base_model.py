#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace base_model 依赖图提取
=================================
从全量清单 jsonl 流式读取，从 Tags 字段提取 base_model 关系：
  - base_model:ORG/MODEL       → 父模型（微调/蒸馏基础）
  - base_model:finetune:ORG/MODEL → 明确微调关系
  - arxiv:XXX                  → 论文引用（可选）

输出（CSV 便于后续分析）：
  - {out}/hf_base_model_edges.csv   边：child, parent, relation(finetune|distill|other), crawled_at
  - {out}/hf_base_model_stats.csv   汇总：parent, parent_owner, children 数, 总下载量

用法:
  python extract_hf_base_model.py --list-file hf_models_all.jsonl [--out-dir DIR]
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding.lower() in ("gbk", "gb2312", "cp936"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("HF_OUTPUT_DIR", BASE_DIR / "modelscope_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

EDGE_FILE = OUTPUT_DIR / "hf_base_model_edges.csv"
STATS_FILE = OUTPUT_DIR / "hf_base_model_stats.csv"

EDGE_FIELDS = ["child", "child_owner", "parent", "parent_owner", "relation", "child_downloads", "child_likes"]
STATS_FIELDS = ["parent", "parent_owner", "children", "total_downloads"]


def parse_tags(tags):
    """从 tags 列表提取 base_model 与 arxiv 关系。"""
    if not tags:
        return []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    rels = []
    for t in tags:
        if not isinstance(t, str):
            continue
        if t.startswith("base_model:"):
            val = t[len("base_model:"):].strip()
            if not val:
                continue
            if val.startswith("finetune:"):
                rels.append(("finetune", val[len("finetune:"):].strip()))
            else:
                rels.append(("other", val))
    return rels


def owner_of(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


def main():
    parser = argparse.ArgumentParser(description="HF base_model 依赖图提取")
    parser.add_argument("--list-file", required=True, help="全量清单 jsonl")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    if args.out_dir:
        global OUTPUT_DIR, EDGE_FILE, STATS_FILE
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        EDGE_FILE = OUTPUT_DIR / "hf_base_model_edges.csv"
        STATS_FILE = OUTPUT_DIR / "hf_base_model_stats.csv"

    edges = []
    with open(args.list_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = rec.get("_raw") or rec
            model_id = raw.get("id") or rec.get("Id") or ""
            downloads = raw.get("downloads") or rec.get("Downloads") or 0
            likes = raw.get("likes") or rec.get("Likes") or 0
            rels = parse_tags(raw.get("tags") or rec.get("Tags"))
            for relation, parent in rels:
                if "/" not in parent:
                    continue  # 忽略纯 id（如 base_model:transformers）
                edges.append({
                    "child": model_id,
                    "child_owner": owner_of(model_id),
                    "parent": parent,
                    "parent_owner": owner_of(parent),
                    "relation": relation,
                    "child_downloads": downloads,
                    "child_likes": likes,
                })

    with open(EDGE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(edges)

    # 父模型汇总
    stats = {}
    for e in edges:
        s = stats.setdefault(e["parent"], {"parent": e["parent"], "parent_owner": e["parent_owner"],
                                           "children": 0, "total_downloads": 0})
        s["children"] += 1
        s["total_downloads"] += int(e["child_downloads"] or 0)
    with open(STATS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(stats.values(), key=lambda x: -x["children"]))

    print(f"[done] 边 {len(edges)} 条，父模型 {len(stats)} 个")
    print(f"  边: {EDGE_FILE}")
    print(f"  汇总: {STATS_FILE}")
    for s in sorted(stats.values(), key=lambda x: -x["children"])[:10]:
        print(f"    {s['parent']}: {s['children']} children, {s['total_downloads']:,} dl")


if __name__ == "__main__":
    main()
