#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 全量清单采集（models / datasets / spaces）
========================================================
复用现有 modelscope_output 目录与 state_*.json 断点模式。
端点使用 hf-mirror.com，分页依赖响应 Link 头中的 cursor。

输出（对齐魔搭 models_all.csv / datasets_all.csv 口径）：
  - modelscope_output/hf_models_all.jsonl
  - modelscope_output/hf_models_all.csv
  - modelscope_output/hf_datasets_all.jsonl/.csv
  - modelscope_output/hf_studios_all.jsonl/.csv
  - modelscope_output/state_hf_models.json 等断点文件

注意：
  - file_size 在列表接口中不可得（siblings 无 size），全量阶段留空；
    子集深挖阶段通过 /api/{type}/{id}/tree/main 补齐。
  - downloads 为 HF 近 30 天计数，与魔搭累计值口径不同，仅用于排序/分位数。
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
# 输出目录：默认魔搭目录；可通过 --out-dir 指定独立 HF 目录（不与魔搭数据混放）
OUTPUT_DIR = Path(os.environ.get("HF_OUTPUT_DIR", BASE_DIR / "modelscope_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

HF_ORIGIN = "https://hf-mirror.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# 资源类型 → API 路径片段、输出文件前缀
RESOURCE_CFG = {
    "models":   {"endpoint": "models",   "prefix": "hf_models",   "repo_type": "model"},
    "datasets": {"endpoint": "datasets", "prefix": "hf_datasets", "repo_type": "dataset"},
    "spaces":   {"endpoint": "spaces",   "prefix": "hf_studios",  "repo_type": "space"},
}

# 对齐魔搭 models_all.csv 的字段顺序，并补充清单要求的 library_name
CSV_FIELDS = [
    "CreatedAt", "Description", "Downloads", "Id", "License", "Likes",
    "Name", "Owner", "RepoType", "Tags", "UpdatedAt", "Visibility",
    "display_name", "file_size", "gated", "login_required", "private", "tasks",
    "library_name",
]

REQUEST_DELAY = 0.8          # 请求间隔（秒）
SAVE_EVERY = 1000            # 每采 1000 条写一次 state


# ============================================================================
# 工具函数
# ============================================================================
def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"count": 0, "next_cursor": None, "finished": False}


def save_state(state_file: Path, state: dict):
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def extract_license(tags: list) -> str:
    """从 HF tags 中提取 license:xxx。"""
    if not tags:
        return ""
    for t in tags:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1].strip()
    return ""


def extract_library_name(tags: list) -> str:
    """从 HF tags 中提取 library:xxx。"""
    if not tags:
        return ""
    for t in tags:
        if isinstance(t, str) and t.startswith("library:"):
            return t.split(":", 1)[1].strip()
    return ""


def normalize_record(item: dict, repo_type: str) -> dict:
    """将 HF 列表接口字段映射为魔搭口径。"""
    _id = item.get("id") or item.get("modelId") or ""
    author = item.get("author") or ""
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    # Visibility：魔搭为数字，HF private 为 bool；这里做近似映射
    private = bool(item.get("private"))
    visibility = 1 if private else 5

    # gated：HF 返回 "auto", "manual", False, True 等
    gated_raw = item.get("gated")
    gated = str(gated_raw).lower() not in ("", "false", "none", "0")
    login_required = gated

    return {
        "CreatedAt": item.get("createdAt", ""),
        "Description": (item.get("description") or "").replace("\n", " ").replace("\r", " "),
        "Downloads": item.get("downloads", 0),
        "Id": _id,
        "License": extract_license(tags),
        "Likes": item.get("likes", 0),
        "Name": item.get("modelId") or _id,
        "Owner": author,
        "RepoType": repo_type,
        "Tags": json.dumps(tags, ensure_ascii=False),
        "UpdatedAt": item.get("lastModified", ""),
        "Visibility": visibility,
        "display_name": item.get("modelId") or _id,
        "file_size": "",        # 全量阶段留空，子集阶段补齐
        "gated": gated,
        "login_required": login_required,
        "private": private,
        "tasks": item.get("pipeline_tag", ""),
        "library_name": item.get("library_name") or extract_library_name(tags),
    }


def extract_next_cursor(link_header: str) -> str | None:
    """从 Link 头中提取 next 关系的 cursor 参数。"""
    if not link_header:
        return None
    # Link: <https://huggingface.co/api/models?...&cursor=XXX>; rel="next"
    for part in link_header.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="([^"]+)"', part)
        if match:
            url, rel = match.groups()
            if rel.lower() == "next":
                parsed = urlparse(url)
                qs = parse_qs(parsed.query)
                cursors = qs.get("cursor")
                if cursors:
                    return cursors[0]
    return None


def write_csv_row(writer: csv.DictWriter, record: dict):
    """统一处理 CSV 中可能出现的列表/对象字段。"""
    row = {}
    for k in CSV_FIELDS:
        v = record.get(k, "")
        if isinstance(v, (list, dict, tuple)):
            v = json.dumps(v, ensure_ascii=False)
        row[k] = v
    writer.writerow(row)


# ============================================================================
# 主流程
# ============================================================================
def crawl_resource(resource: str, limit: int = 1000, max_items: int | None = None):
    cfg = RESOURCE_CFG[resource]
    endpoint = cfg["endpoint"]
    prefix = cfg["prefix"]
    repo_type = cfg["repo_type"]

    jsonl_file = OUTPUT_DIR / f"{prefix}_all.jsonl"
    csv_file = OUTPUT_DIR / f"{prefix}_all.csv"
    state_file = OUTPUT_DIR / f"state_hf_{resource}.json"
    state = load_state(state_file)

    if state.get("finished"):
        print(f"[{resource}] 已完成，跳过。如需重跑请删除 {state_file}")
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    # 准备 CSV（断点续传时追加，不重复写 header）
    csv_exists = csv_file.exists() and csv_file.stat().st_size > 0
    csv_f = open(csv_file, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if not csv_exists:
        writer.writeheader()

    jsonl_f = open(jsonl_file, "a", encoding="utf-8")

    cursor = state.get("next_cursor")
    count = state.get("count", 0)
    page = 0

    try:
        while True:
            if cursor:
                url = f"{HF_ORIGIN}/api/{endpoint}?full=true&limit={limit}&cursor={cursor}"
            else:
                url = f"{HF_ORIGIN}/api/{endpoint}?full=true&limit={limit}"

            try:
                r = session.get(url, timeout=60)
            except requests.RequestException as e:
                print(f"[{resource}] 网络错误: {e}; 30 秒后重试...")
                time.sleep(30)
                continue

            if r.status_code == 429:
                wait = min(2 ** (page % 6) * 5, 300)
                print(f"[{resource}] 429 限流，等待 {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"[{resource}] HTTP {r.status_code}: {url[:120]}")
                print(f"  body: {r.text[:200]}")
                break

            items = r.json()
            if not isinstance(items, list) or not items:
                print(f"[{resource}] 空响应，结束。count={count}")
                state["finished"] = True
                break

            for item in items:
                record = normalize_record(item, repo_type)
                raw_line = {
                    **record,
                    "_raw": item,
                    "crawled_at": time.time(),
                }
                jsonl_f.write(json.dumps(raw_line, ensure_ascii=False) + "\n")
                write_csv_row(writer, record)
                count += 1

                if max_items and count >= max_items:
                    print(f"[{resource}] 达到 max_items={max_items}，暂停。")
                    state["count"] = count
                    state["next_cursor"] = cursor
                    return

            jsonl_f.flush()
            csv_f.flush()

            if count % SAVE_EVERY < len(items):
                state["count"] = count
                state["next_cursor"] = cursor
                save_state(state_file, state)
                print(f"[{resource}] progress: {count} saved")

            next_cursor = extract_next_cursor(r.headers.get("Link", ""))
            if not next_cursor:
                print(f"[{resource}] 无下一页 cursor，结束。count={count}")
                state["finished"] = True
                break

            cursor = next_cursor
            page += 1
            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print(f"\n[{resource}] 用户中断，已保存断点。")
    finally:
        state["count"] = count
        state["next_cursor"] = cursor
        save_state(state_file, state)
        jsonl_f.close()
        csv_f.close()
        print(f"[{resource}] 结束，累计 {count} 条。")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 全量清单采集")
    parser.add_argument(
        "resource", choices=["models", "datasets", "spaces", "all"],
        help="要采集的资源类型"
    )
    parser.add_argument("--limit", type=int, default=1000, help="每页条数（默认 1000）")
    parser.add_argument("--max-items", type=int, default=None, help="最多采集条数（调试用）")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录（默认 HF_OUTPUT_DIR 环境变量或 modelscope_output）")
    args = parser.parse_args()

    if args.out_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    resources = ["models", "datasets", "spaces"] if args.resource == "all" else [args.resource]
    for res in resources:
        crawl_resource(res, limit=args.limit, max_items=args.max_items)


if __name__ == "__main__":
    main()
