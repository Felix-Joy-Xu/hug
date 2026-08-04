#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 子集行为数据采集（主流方案：huggingface_hub）
========================================================
采集内容：
  - commits:      HfApi().list_repo_commits()
  - discussions:  REST /api/models/{id}/discussions
  - README:       hf_hub_download()
  - file tree:    HfApi().list_repo_files()

输出：
  - modelscope_output/hf_commit_history.jsonl
  - modelscope_output/hf_comments_all.jsonl
  - modelscope_output/hf_model_cards/{safe_id}_README.md
  - modelscope_output/hf_model_dependencies.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "modelscope_output"
OUTPUT_DIR.mkdir(exist_ok=True)

HF_BASE = os.environ.get("HF_BASE", "https://huggingface.co").rstrip("/")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

MODELS_FILE = OUTPUT_DIR / "hf_models_all.jsonl"

COMMIT_FILE = OUTPUT_DIR / "hf_commit_history.jsonl"
COMMENT_FILE = OUTPUT_DIR / "hf_comments_all.jsonl"
DEP_FILE = OUTPUT_DIR / "hf_model_dependencies.jsonl"
CARD_DIR = OUTPUT_DIR / "hf_model_cards"
CARD_DIR.mkdir(exist_ok=True)

STATE_FILE = OUTPUT_DIR / "state_hf_subset.json"

REQUEST_DELAY = 0.5
SAVE_EVERY = 20


# ============================================================================
# 工具函数
# ============================================================================
def safe_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_")


def load_jsonl(path: Path) -> list:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "count": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fetch_json(session: requests.Session, url: str, max_retries: int = 3) -> tuple:
    """抓取 JSON，返回 (data, status)。"""
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                try:
                    return r.json(), 200
                except Exception:
                    return r.text, 200
            if r.status_code in (429, 503):
                time.sleep(min(2 ** attempt * 3, 30))
                continue
            return None, r.status_code
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return str(e), -1
    return None, -1


# ============================================================================
# 采集函数
# ============================================================================
def fetch_model_data(api: HfApi, session: requests.Session, model_id: str) -> dict:
    """采集单个模型的行为数据。"""
    result = {
        "model_id": model_id,
        "crawled_at": time.time(),
        "commits": {"status": "pending"},
        "discussions": {"status": "pending"},
        "readme": {"status": "pending"},
        "tree": {"status": "pending"},
    }

    # 1. commits（官方库）
    try:
        commits = api.list_repo_commits(model_id)
        commit_list = []
        for c in commits:
            commit_list.append({
                "commit_id": c.commit_id,
                "message": c.title,
                "created_at": c.created_at.isoformat() if c.created_at else "",
                "authors": c.authors,
            })
        result["commits"] = {"status": "success", "count": len(commit_list), "data": commit_list}
    except Exception as e:
        result["commits"] = {"status": "error", "error": str(e)}

    # 2. discussions / comments（REST）
    disc_url = f"{HF_BASE}/api/models/{model_id}/discussions"
    disc_data, disc_status = fetch_json(session, disc_url)
    result["discussions"] = {
        "status": "success" if disc_status == 200 else "error",
        "http_status": disc_status,
        "url": disc_url,
        "data": disc_data if disc_status == 200 else None,
    }

    # 3. README（官方库）
    try:
        readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_text = f.read()
        result["readme"] = {"status": "success", "length": len(readme_text)}
        with open(CARD_DIR / f"{safe_id(model_id)}_README.md", "w", encoding="utf-8") as f:
            f.write(readme_text)
    except Exception as e:
        result["readme"] = {"status": "error", "error": str(e)}

    # 4. tree / file_size（官方库）
    try:
        files = api.list_repo_files(model_id)
        file_size = 0
        file_count = 0
        for f in files:
            if isinstance(f, dict) and f.get("type") == "file":
                file_count += 1
                file_size += f.get("size", 0)
        result["tree"] = {
            "status": "success",
            "file_count": file_count,
            "file_size": file_size,
        }
    except Exception as e:
        result["tree"] = {"status": "error", "error": str(e)}

    return result


def write_dep_record(model_id: str, tree_result: dict):
    """写依赖/文件树记录。"""
    record = {
        "model_id": model_id,
        "crawled_at": time.time(),
        "file_count": tree_result.get("file_count", 0),
        "file_size": tree_result.get("file_size", 0),
    }
    with open(DEP_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="HuggingFace 子集行为数据采集")
    parser.add_argument("--total", type=int, default=200, help="子集总数（默认 200）")
    parser.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    args = parser.parse_args()

    if not MODELS_FILE.exists():
        print(f"[error] 未找到 {MODELS_FILE}，请先运行 crawl_hf_full.py")
        sys.exit(1)

    # 按下载量取 Top N
    all_models = load_jsonl(MODELS_FILE)
    models = []
    for rec in all_models:
        raw = rec.get("_raw") or rec
        if raw.get("id"):
            models.append(raw)
    models.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    subset = models[:args.total]
    print(f"[main] 从 {len(models)} 个模型中选取 Top {len(subset)} 个（按下载量）")

    state = load_state()
    completed = set(state.get("completed", []))
    todo = [m for m in subset if m.get("id") not in completed]
    print(f"[main] 已完成 {len(completed)}，待采集 {len(todo)}")

    commit_f = open(COMMIT_FILE, "a", encoding="utf-8")
    comment_f = open(COMMENT_FILE, "a", encoding="utf-8")

    session = requests.Session()
    session.headers.update(HEADERS)
    api = HfApi()

    done_this_run = 0
    try:
        for m in todo:
            model_id = m["id"]
            try:
                result = fetch_model_data(api, session, model_id)
            except Exception as e:
                result = {
                    "model_id": model_id,
                    "crawled_at": time.time(),
                    "status": "error",
                    "error": str(e),
                }

            # commits
            commit_record = {
                "model_id": model_id,
                "crawled_at": result["crawled_at"],
                "result": result["commits"],
            }
            commit_f.write(json.dumps(commit_record, ensure_ascii=False) + "\n")

            # discussions / comments
            comment_record = {
                "model_id": model_id,
                "crawled_at": result["crawled_at"],
                "result": result["discussions"],
            }
            comment_f.write(json.dumps(comment_record, ensure_ascii=False) + "\n")

            # tree / dependencies
            write_dep_record(model_id, result["tree"])

            completed.add(model_id)
            done_this_run += 1

            if done_this_run % SAVE_EVERY == 0:
                state["completed"] = sorted(completed)
                state["count"] = len(completed)
                save_state(state)
                print(f"[progress] 本运行 +{done_this_run}，累计 {len(completed)}/{len(subset)}")

            time.sleep(REQUEST_DELAY)
    except KeyboardInterrupt:
        print("\n[main] 用户中断")
    finally:
        state["completed"] = sorted(completed)
        state["count"] = len(completed)
        save_state(state)
        commit_f.close()
        comment_f.close()
        print(f"[main] 结束，本运行 +{done_this_run}，累计完成 {len(completed)}/{len(subset)}")


if __name__ == "__main__":
    main()
