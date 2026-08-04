#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 组织 + 用户画像采集
=================================
从全量清单（hf_models_all.jsonl 等）提取所有 owner，逐个抓取：
  - 个人:   GET /api/users/{name}/overview   (isPro, numModels, numDatasets, numSpaces,
                                               numDiscussions, numPapers, numLikes, numFollowers, orgs, createdAt)
  - 组织:   GET /api/organizations/{name}/overview (numUsers, numModels, numSpaces, numDatasets,
                                                    numPapers, numFollowers, plan, fullname)
判断规则：/api/users/{name}/overview 返回 200 为个人，404 为组织（再查 org overview）。

输出（对齐魔搭 meta_orgs.json / meta_models.json 口径）：
  - {out}/hf_org_profiles.jsonl   组织画像
  - {out}/hf_user_profiles.jsonl  个人画像
  - {out}/hf_owner_index.csv      全部 owner 汇总（类型 + 主要指标）
状态: {out}/state_hf_profiles.json（已完成的 owner，断点续爬）

用法:
  python crawl_hf_profiles.py --list-file hf_models_all.jsonl [--workers 4] [--out-dir DIR]
  python crawl_hf_profiles.py --list-file owners.txt [--workers 4]
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

if sys.stdout.encoding.lower() in ("gbk", "gb2312", "cp936"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("HF_OUTPUT_DIR", BASE_DIR / "modelscope_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

HF_BASE = os.environ.get("HF_BASE", "https://hf-mirror.com")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

REQUEST_DELAY = 0.3
SAVE_EVERY = 200
WRITE_LOCK = threading.Lock()

ORG_FILE = OUTPUT_DIR / "hf_org_profiles.jsonl"
USER_FILE = OUTPUT_DIR / "hf_user_profiles.jsonl"
INDEX_FILE = OUTPUT_DIR / "hf_owner_index.csv"
STATE_FILE = OUTPUT_DIR / "state_hf_profiles.json"

INDEX_FIELDS = [
    "name", "type", "fullname", "plan", "isPro", "isVerified",
    "numModels", "numDatasets", "numSpaces", "numFollowers", "numLikes",
    "numDiscussions", "numPapers", "numUsers", "createdAt", "crawled_at",
]

ORG_META_FIELDS = ["_id", "avatarUrl", "fullname", "name", "isVerified", "isFollowing", "plan",
                   "numUsers", "numModels", "numSpaces", "numDatasets", "numKernels",
                   "numBuckets", "numPapers", "numFollowers"]
USER_META_FIELDS = ["_id", "avatarUrl", "isPro", "fullname", "numModels", "numDatasets",
                    "numSpaces", "numKernels", "numBuckets", "numDiscussions", "numPapers",
                    "numUpvotes", "numLikes", "numFollowers", "numFollowing", "numFollowingOrgs",
                    "orgs", "user", "type", "isFollowing", "details", "createdAt"]


# ============================================================================
# 工具函数
# ============================================================================
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "count": 0, "orgs": 0, "users": 0, "errors": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def extract_owners(list_file: Path) -> list:
    """从全量清单 jsonl（每行含 author/Owner）或 txt（每行一个 owner）提取去重 owner。"""
    owners = set()
    is_jsonl = list_file.suffix.lower() == ".jsonl"
    with open(list_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 兼容新格式（raw）与旧格式（_raw.author）
                author = rec.get("author") or rec.get("Owner") or (rec.get("_raw") or {}).get("author")
            else:
                author = line
            if author:
                owners.add(str(author).strip())
    return sorted(owners)


def fetch_json(session, url, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code in (429, 503):
                time.sleep(min(2 ** attempt * 3, 30))
                continue
            if r.status_code == 200:
                return r.json(), 200
            return None, r.status_code
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, -1
    return None, -1


def fetch_profile(session, owner: str):
    """抓取个人或组织画像。返回 (record, owner_type) 或 (None, error_type)。"""
    # 1. 先试个人
    data, status = fetch_json(session, f"{HF_BASE}/api/users/{quote(owner)}/overview")
    if status == 200 and data:
        return {
            "name": owner,
            "type": "user",
            "crawled_at": time.time(),
            "meta": {k: data.get(k) for k in USER_META_FIELDS},
        }, "user"

    # 2. 404 → 可能是组织
    if status == 404:
        data, status = fetch_json(session, f"{HF_BASE}/api/organizations/{quote(owner)}/overview")
        if status == 200 and data:
            return {
                "name": owner,
                "type": "org",
                "crawled_at": time.time(),
                "meta": {k: data.get(k) for k in ORG_META_FIELDS},
            }, "org"

    return None, f"http{status}"


def append_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_index(writer, record: dict):
    meta = record.get("meta", {})
    row = {
        "name": record.get("name"),
        "type": record.get("type"),
        "fullname": meta.get("fullname", ""),
        "plan": meta.get("plan", ""),
        "isPro": meta.get("isPro", ""),
        "isVerified": meta.get("isVerified", ""),
        "numModels": meta.get("numModels", ""),
        "numDatasets": meta.get("numDatasets", ""),
        "numSpaces": meta.get("numSpaces", ""),
        "numFollowers": meta.get("numFollowers", ""),
        "numLikes": meta.get("numLikes", ""),
        "numDiscussions": meta.get("numDiscussions", ""),
        "numPapers": meta.get("numPapers", ""),
        "numUsers": meta.get("numUsers", ""),
        "createdAt": meta.get("createdAt", ""),
        "crawled_at": record.get("crawled_at", ""),
    }
    writer.writerow({k: row.get(k, "") for k in INDEX_FIELDS})


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 组织/用户画像采集")
    parser.add_argument("--list-file", required=True, help="全量清单 jsonl 或 owner 列表 txt")
    parser.add_argument("--workers", type=int, default=1, help="并发数（默认 1，低调防限流）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 个 owner（调试用）")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录（默认 HF_OUTPUT_DIR 或 modelscope_output）")
    args = parser.parse_args()

    if args.out_dir:
        global OUTPUT_DIR, ORG_FILE, USER_FILE, INDEX_FILE, STATE_FILE
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        ORG_FILE = OUTPUT_DIR / "hf_org_profiles.jsonl"
        USER_FILE = OUTPUT_DIR / "hf_user_profiles.jsonl"
        INDEX_FILE = OUTPUT_DIR / "hf_owner_index.csv"
        STATE_FILE = OUTPUT_DIR / "state_hf_profiles.json"

    list_path = Path(args.list_file)
    owners = extract_owners(list_path)
    if args.limit:
        owners = owners[:args.limit]
    print(f"[main] 共 {len(owners)} 个唯一 owner")

    state = load_state()
    completed = set(state.get("completed", []))
    todo = [o for o in owners if o not in completed]
    print(f"[main] 已完成 {len(completed)}，待采集 {len(todo)}")

    if not todo:
        print("[main] 全部完成")
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    index_exists = INDEX_FILE.exists() and INDEX_FILE.stat().st_size > 0
    index_f = open(INDEX_FILE, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(index_f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
    if not index_exists:
        writer.writeheader()

    def process(owner):
        """抓单个 owner；返回 (owner, record, owner_type) 或 (owner, None, err)。"""
        try:
            record, owner_type = fetch_profile(session, owner)
        except Exception as e:
            return owner, None, f"err:{type(e).__name__}"
        return owner, record, owner_type

    done = 0
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for owner, record, owner_type in pool.map(process, todo):
            if record:
                if owner_type == "org":
                    append_jsonl(ORG_FILE, record)
                else:
                    append_jsonl(USER_FILE, record)
                append_index(writer, record)
                if owner_type == "org":
                    state["orgs"] += 1
                else:
                    state["users"] += 1
            else:
                state["errors"] += 1
                if state["errors"] <= 20:
                    print(f"  [{owner}] 抓取失败: {owner_type}")

            completed.add(owner)
            state["completed"] = sorted(completed)
            state["count"] = len(completed)
            done += 1

            if done % SAVE_EVERY == 0:
                with WRITE_LOCK:
                    index_f.flush()
                    save_state(state)
                print(f"[progress] +{done}，累计 {len(completed)}/{len(owners)} "
                      f"(orgs={state['orgs']}, users={state['users']}, err={state['errors']})")

    index_f.flush()
    index_f.close()
    save_state(state)
    print(f"[done] 累计 {len(completed)}/{len(owners)} "
          f"(orgs={state['orgs']}, users={state['users']}, err={state['errors']})")
    print(f"  组织画像: {ORG_FILE}")
    print(f"  个人画像: {USER_FILE}")
    print(f"  owner 汇总: {INDEX_FILE}")


if __name__ == "__main__":
    main()
