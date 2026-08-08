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
# 多 token 轮询：HF_TOKENS 逗号分隔；HF_TOKEN_INDEX 指定固定用第 N 个（每 job 专用一个）
HF_TOKENS = [t.strip() for t in os.environ.get("HF_TOKENS", os.environ.get("HF_TOKEN", "")).split(",") if t.strip()]
HF_TOKEN_INDEX = int(os.environ.get("HF_TOKEN_INDEX", "-1"))
if HF_TOKEN_INDEX >= 0 and HF_TOKEN_INDEX < len(HF_TOKENS):
    HF_TOKENS = [HF_TOKENS[HF_TOKEN_INDEX]]  # 只用一个指定 token
    print(f"[config] 使用第 {HF_TOKEN_INDEX} 个 HF token（专用配额 200/min）", flush=True)
elif HF_TOKENS:
    print(f"[config] 使用 {len(HF_TOKENS)} 个 HF token 轮询（总配额 {len(HF_TOKENS)*200}/min 左右）", flush=True)
else:
    print("[config] 未提供 HF token，使用匿名访问（限流较严）", flush=True)

REQUEST_DELAY = 0.3
SAVE_EVERY = 200
WRITE_LOCK = threading.Lock()
# 云端定期回写：每 CHECKPOINT_MINUTES 分钟生成进度报告并 git push 断点到仓库（防超时丢数据）
CHECKPOINT_MINUTES = 30
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
_last_commit_count = [0]
_last_checkpoint_ts = [0.0]


def load_shard_state(shard: int, out_dir=None) -> dict:
    """读取指定 shard 的 state 文件（不存在返回空）。"""
    d = Path(out_dir) if out_dir else OUTPUT_DIR
    sf = d / f"state_hf_profiles_{shard}.json"
    if not sf.exists():
        return {}
    try:
        with open(sf, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_progress_report(shards: int, started_ts: float, out_dir=None):
    """生成单一汇总进度报告 PROGRESS.md（读全部 shard 的 state）。"""
    d = Path(out_dir) if out_dir else OUTPUT_DIR
    report = d / "PROGRESS.md"
    states = []
    totals = []
    for i in range(shards):
        s = load_shard_state(i, out_dir=out_dir)
        states.append(s)
        totals.append(s.get("total", 0) if s else 0)
    done_total = sum(s.get("count", 0) for s in states)
    org_total = sum(s.get("orgs", 0) for s in states)
    user_total = sum(s.get("users", 0) for s in states)
    err_total = sum(s.get("errors", 0) for s in states)
    all_total = sum(totals)
    pct = done_total / all_total * 100 if all_total else 0
    elapsed = time.time() - started_ts
    rate = done_total / (elapsed / 60) if elapsed > 0 else 0
    remain = (all_total - done_total) / rate if rate > 0 else 0

    lines = [
        "# HuggingFace 组织/用户画像采集进度",
        "",
        f"- **更新时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
        f"- **分片数**: {shards}",
        f"- **总体进度**: {done_total:,} / {all_total:,} 个 owner（{pct:.1f}%）",
        f"- **组织数**: {org_total:,}",
        f"- **个人用户数**: {user_total:,}",
        f"- **失败数**: {err_total:,}",
        f"- **采集速率**: {rate:.0f} 个/分钟",
        f"- **已运行**: {elapsed/3600:.1f} 小时",
        f"- **预计剩余**: {remain/60:.1f} 小时",
        "",
        "## 各分片进度",
        "",
    ]
    for i in range(shards):
        s = states[i]
        if not s:
            lines.append(f"- 分片 {i}: 尚未开始")
            continue
        t = totals[i]
        p = s.get("count", 0) / t * 100 if t else 0
        lines.append(f"- 分片 {i}: {s.get('count', 0):,} / {t:,}（{p:.1f}%），"
                     f"orgs={s.get('orgs', 0):,}，users={s.get('users', 0):,}，err={s.get('errors', 0):,}")
    lines += [
        "",
        "## 数据文件（全部在 modelscope_output/ 下）",
        "",
        "- `hf_org_profiles_{N}.jsonl`：各分片组织画像",
        "- `hf_user_profiles_{N}.jsonl`：各分片个人画像",
        "- `hf_owner_index_{N}.csv`：各分片 owner 汇总",
        "",
        "> 本报告由爬虫每 30 分钟自动生成并提交。",
    ]
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


def _git(cmd, cwd, timeout=60):
    """带超时的 git 命令，返回 (returncode, stderr)。"""
    import subprocess
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)[:100]


def checkpoint_commit(state: dict, total: int, started_ts: float):
    """每 30 分钟生成汇总进度报告并保存断点（git push 由 workflow 后台守护进程负责，避免阻塞采集）。"""
    if not IS_GITHUB_ACTIONS:
        return
    try:
        report = write_progress_report(_shards, started_ts)
        done = state.get("count", 0)
        print(f"[checkpoint] 断点与报告已更新（{done}/{total}），报告: {report.name}", flush=True)
    except Exception as e:
        print(f"[checkpoint] 失败: {str(e)[:80]}", flush=True)

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


def _get_token():
    """轮询返回下一个 token；无 token 返回 None。"""
    if not HF_TOKENS:
        return None
    idx = int(time.time() * 100) % len(HF_TOKENS)
    return HF_TOKENS[idx]


# ============================================================
# 全局令牌桶限速：控制本 job 的请求速率，
# 5 个 job 共享同一 HF 账号配额（200 req/min），
# 每 job 限速 RPS_PER_JOB，避免 429 限流。
# ============================================================
class TokenBucket:
    def __init__(self, rps: float):
        self.capacity = max(1.0, rps)  # 桶容量至少 1
        self.tokens = 1.0
        self.rps = rps
        self.last = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rps)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rps
            time.sleep(wait)


RPS_PER_JOB = float(os.environ.get("HF_RPS", "0.67"))  # 默认 40 req/min/job
_bucket = TokenBucket(RPS_PER_JOB)


def fetch_json(session, url, retries=6):
    for attempt in range(retries):
        try:
            _bucket.acquire()  # 全局限速
            headers = dict(session.headers)
            tok = _get_token()
            if tok:
                headers["Authorization"] = f"Bearer {tok}"
            r = session.get(url, headers=headers, timeout=30)
            if r.status_code in (429, 503):
                # 按 Retry-After 头等待（限流核心应对）
                ra = r.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else min(2 ** attempt * 5, 60)
                time.sleep(wait)
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

    # 3. 返回错误类型（429=限流，404=用户/组织均不存在，其他）
    if status in (429, 503):
        return None, "ratelimit"
    if status == 404:
        return None, "notfound"
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
    parser.add_argument("--shard", type=int, default=0, help="分片索引（0-based）")
    parser.add_argument("--shards", type=int, default=1, help="总片数")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 个 owner（调试用）")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录（默认 HF_OUTPUT_DIR 或 modelscope_output）")
    args = parser.parse_args()

    global _shard, _shards
    _shard = args.shard
    _shards = args.shards

    if args.out_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 输出文件始终带 shard 后缀（shards>1 时），保证多 job 互不冲突
    global ORG_FILE, USER_FILE, INDEX_FILE, STATE_FILE
    suf = f"_{args.shard}" if args.shards > 1 else ""
    ORG_FILE = OUTPUT_DIR / f"hf_org_profiles{suf}.jsonl"
    USER_FILE = OUTPUT_DIR / f"hf_user_profiles{suf}.jsonl"
    INDEX_FILE = OUTPUT_DIR / f"hf_owner_index{suf}.csv"
    STATE_FILE = OUTPUT_DIR / f"state_hf_profiles{suf}.json"

    list_path = Path(args.list_file)
    owners_all = extract_owners(list_path)
    total_all = len(owners_all)
    if args.limit:
        owners = owners_all[:args.limit]
    else:
        owners = owners_all
    # 分片：按 shards 均分
    if args.shards > 1:
        per = (total_all + args.shards - 1) // args.shards
        owners = owners_all[args.shard * per:(args.shard + 1) * per]
    print(f"[main] 片 {args.shard}/{args.shards}：本片 {len(owners)} 个 owner（全量 {total_all}）", flush=True)

    state = load_state()
    state["total"] = len(owners)  # 供汇总进度报告使用
    completed = set(state.get("completed", []))
    todo = [o for o in owners if o not in completed]
    print(f"[main] 已完成 {len(completed)}，待采集 {len(todo)}", flush=True)

    if not todo:
        print("[main] 全部完成", flush=True)
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    index_exists = INDEX_FILE.exists() and INDEX_FILE.stat().st_size > 0
    index_f = open(INDEX_FILE, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(index_f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
    if not index_exists:
        writer.writeheader()

    done = 0
    workers = max(1, args.workers)
    started_ts = time.time()
    _last_checkpoint_ts[0] = started_ts
    # 时间上限：到点主动收尾退出（防云端 360 分钟超时取消丢数据）
    max_minutes = float(os.environ.get("MAX_MINUTES", "0") or 0)
    time_up = False

    def process(owner):
        try:
            record, owner_type = fetch_profile(session, owner)
        except Exception as e:
            return owner, None, f"err:{type(e).__name__}"
        return owner, record, owner_type

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, o) for o in todo]
        for fut in as_completed(futures):
            if time_up:
                fut.cancel()
                continue
            try:
                owner, record, owner_type = fut.result()
            except Exception:
                continue
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
                err_type = owner_type if isinstance(owner_type, str) else str(owner_type)
                state.setdefault("err_detail", {})
                state["err_detail"][err_type] = state["err_detail"].get(err_type, 0) + 1
                if state["errors"] <= 20:
                    print(f"  [{owner}] 抓取失败: {owner_type}", flush=True)

            completed.add(owner)
            state["completed"] = sorted(completed)
            state["count"] = len(completed)
            done += 1

            if done % SAVE_EVERY == 0:
                with WRITE_LOCK:
                    index_f.flush()
                    save_state(state)
                print(f"[progress] +{done}，累计 {len(completed)}/{len(owners)} "
                      f"(orgs={state['orgs']}, users={state['users']}, err={state['errors']})", flush=True)

            # 云端每 30 分钟生成进度报告并回写断点（防超时丢数据）
            if time.time() - _last_checkpoint_ts[0] >= CHECKPOINT_MINUTES * 60:
                _last_checkpoint_ts[0] = time.time()
                checkpoint_commit(state, len(owners), started_ts)

            # 时间上限：主动收尾（保存断点并正常退出，job 标记成功）
            if max_minutes and (time.time() - started_ts) >= max_minutes * 60:
                time_up = True
                print(f"[main] 达到时间上限 {max_minutes} 分钟，主动收尾（已处理 {done}）", flush=True)
                break

    index_f.flush()
    index_f.close()
    save_state(state)
    print(f"[done] 累计 {len(completed)}/{len(owners)} "
          f"(orgs={state['orgs']}, users={state['users']}, err={state['errors']})", flush=True)
    print(f"  组织画像: {ORG_FILE}", flush=True)
    print(f"  个人画像: {USER_FILE}", flush=True)
    print(f"  owner 汇总: {INDEX_FILE}", flush=True)
    # 结束时提交最终进度报告
    checkpoint_commit(state, len(owners), started_ts)


if __name__ == "__main__":
    main()
