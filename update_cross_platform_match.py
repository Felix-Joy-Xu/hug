#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF ↔ 魔搭跨平台匹配表更新
==========================
输入：
  - modelscope_output/models_all.json（魔搭全量）
  - modelscope_output/hf_models_all.jsonl（HF 全量，由 crawl_hf_full.py 生成）

输出：
  - modelscope_output/cross_platform_match_full.json
  - modelscope_output/cross_platform_match_full.csv

匹配策略（按优先级）：
  1. model_id 直接相等（大小写不敏感）。
  2. owner 别名映射 + name 相等（如 THUDM ↔ ZhipuAI）。
  3. name 部分相等（处理大小写、下划线/连字符差异）。
  4. 对仍未匹配项做名称相似度辅助（Jaccard / Levenshtein），供人工抽检。

新增字段：
  - platform_presence: both / hf_only / ms_only
  - match_method: exact / owner_alias / name / fuzzy / manual
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "modelscope_output"

MS_FILE = OUTPUT_DIR / "models_all.json"
HF_FILE = OUTPUT_DIR / "hf_models_all.jsonl"

OUT_JSON = OUTPUT_DIR / "cross_platform_match_full.json"
OUT_CSV = OUTPUT_DIR / "cross_platform_match_full.csv"

# owner 别名映射（已知或常见的国产模型 owner 差异）
OWNER_ALIASES = {
    "zhipuai": ["thudm"],
    "thudm": ["zhipuai"],
    "01ai": ["01-ai"],
    "01-ai": ["01ai"],
    "llm-research": ["meta-llama", "mistralai", "google", "microsoft"],
    "ai-modelscope": ["google", "tencent", "openai", "stabilityai"],
}

# CSV 字段
CSV_FIELDS = [
    "label", "msap_id", "hf_id", "platform_presence", "match_method",
    "has_hf_counterpart", "hf_gated", "downloads_msap", "downloads_hf",
    "download_ratio_hf_over_msap", "license_msap", "license_hf", "license_match",
    "hf_readme_length", "msap_readme_length", "tags_hf", "last_modified_hf", "created_at_hf",
]


# ============================================================================
# 工具函数
# ============================================================================
def load_ms_models(path: Path) -> dict:
    """加载魔搭全量模型，返回 {lowercase_id: model_dict}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for m in data:
        mid = m.get("Id") or m.get("id") or ""
        if mid:
            result[mid.lower()] = m
    return result


def load_hf_models(path: Path) -> dict:
    """加载 HF 全量模型 jsonl，返回 {lowercase_id: raw_model_dict}。"""
    result = {}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = rec.get("_raw") or rec
            mid = raw.get("id") or raw.get("modelId") or ""
            if mid:
                result[mid.lower()] = raw
    return result


def parse_id(model_id: str):
    """拆分 owner 和 name，统一小写。"""
    parts = model_id.split("/")
    if len(parts) == 2:
        return parts[0].lower(), parts[1].lower()
    return "", model_id.lower()


def normalize_name(name: str) -> str:
    """归一化模型名用于模糊匹配：小写、去掉常见后缀、统一分隔符。"""
    name = name.lower()
    # 去掉常见量化/格式后缀
    for suffix in ["-gguf", "-gptq", "-awq", "-bnb", "-4bit", "-8bit", "-fp16", "-bf16"]:
        name = name.split(suffix)[0]
    name = re.sub(r"[-_.]+", "-", name).strip("-")
    return name


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(a.split("-")), set(b.split("-"))
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def ratio_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def extract_license_ms(m: dict) -> str:
    lic = m.get("License") or m.get("license") or ""
    return str(lic).lower().strip()


def extract_license_hf(m: dict) -> str:
    tags = m.get("tags") or []
    for t in tags:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1].lower().strip()
    card = m.get("cardData") or {}
    lic = card.get("license") or ""
    return str(lic).lower().strip()


# ============================================================================
# 匹配逻辑
# ============================================================================
def _norm_and_tokens(model_id: str):
    """返回归一化 name 和 token 集合。"""
    _, name = parse_id(model_id)
    norm = normalize_name(name)
    return norm, set(norm.split("-")) if norm else set()


def _fuzzy_worker(args):
    """多进程 worker：对一批 MS 模型做模糊匹配。"""
    ms_batch, hf_index, threshold = args
    local_candidates = []
    for ms_key, ms in ms_batch:
        _, ms_name = parse_id(ms.get("Id") or ms.get("id") or "")
        ms_norm, ms_tokens = _norm_and_tokens(ms_name)
        if not ms_tokens:
            continue
        # 只和共享 token 的 HF 候选比较
        hf_candidates = set()
        for tok in ms_tokens:
            hf_candidates.update(hf_index.get(tok, []))
        for hf_key, hf in hf_candidates:
            _, hf_name = parse_id(hf.get("id") or hf.get("modelId") or "")
            hf_norm, _ = _norm_and_tokens(hf_name)
            score = ratio_similarity(ms_norm, hf_norm)
            if score >= threshold:
                local_candidates.append((score, ms_key, hf_key, ms, hf))
    return local_candidates


def build_match_table(ms_models: dict, hf_models: dict, fuzzy_threshold: float = 0.85,
                      fuzzy_low_threshold: float = 0.75, workers: int = 4):
    """构建匹配表（优化版）。"""
    # 1. 直接匹配
    exact_matches = []
    matched_ms = set()
    matched_hf = set()

    for ms_key, ms in ms_models.items():
        if ms_key in hf_models:
            hf = hf_models[ms_key]
            exact_matches.append((ms, hf, "exact"))
            matched_ms.add(ms_key)
            matched_hf.add(ms_key)

    print(f"[match] exact matches: {len(exact_matches)}")

    # 2. owner 别名 + name 匹配
    alias_matches = []
    ms_by_name = defaultdict(list)
    for key, m in ms_models.items():
        if key in matched_ms:
            continue
        owner, name = parse_id(m.get("Id") or m.get("id") or "")
        ms_by_name[name].append((key, owner, m))

    hf_by_name = defaultdict(list)
    for key, m in hf_models.items():
        if key in matched_hf:
            continue
        owner, name = parse_id(m.get("id") or m.get("modelId") or "")
        hf_by_name[name].append((key, owner, m))

    for name, ms_list in ms_by_name.items():
        if name not in hf_by_name:
            continue
        for ms_key, ms_owner, ms in ms_list:
            for hf_key, hf_owner, hf in hf_by_name[name]:
                aliases = OWNER_ALIASES.get(ms_owner, [])
                if ms_owner == hf_owner or hf_owner in aliases:
                    alias_matches.append((ms, hf, "owner_alias"))
                    matched_ms.add(ms_key)
                    matched_hf.add(hf_key)
                    break

    print(f"[match] alias matches: {len(alias_matches)}")

    # 3. name 归一化匹配：按归一化 name 分组，避免 O(n^2)
    name_matches = []
    remaining_ms = {k: v for k, v in ms_models.items() if k not in matched_ms}
    remaining_hf = {k: v for k, v in hf_models.items() if k not in matched_hf}

    ms_by_norm = defaultdict(list)
    for key, m in remaining_ms.items():
        _, name = parse_id(m.get("Id") or m.get("id") or "")
        ms_by_norm[normalize_name(name)].append((key, m))

    hf_by_norm = defaultdict(list)
    for key, m in remaining_hf.items():
        _, name = parse_id(m.get("id") or m.get("modelId") or "")
        hf_by_norm[normalize_name(name)].append((key, m))

    used_ms = set()
    used_hf = set()
    for norm, ms_list in ms_by_norm.items():
        hf_list = hf_by_norm.get(norm, [])
        if not hf_list:
            continue
        #  popularity 降序，贪心一对一
        ms_list_sorted = sorted(ms_list, key=lambda x: x[1].get("Downloads", 0) or x[1].get("downloads", 0), reverse=True)
        hf_list_sorted = sorted(hf_list, key=lambda x: x[1].get("downloads", 0), reverse=True)
        for ms_key, ms in ms_list_sorted:
            if ms_key in used_ms:
                continue
            for hf_key, hf in hf_list_sorted:
                if hf_key in used_hf:
                    continue
                used_ms.add(ms_key)
                used_hf.add(hf_key)
                name_matches.append((ms, hf, "name"))
                matched_ms.add(ms_key)
                matched_hf.add(hf_key)
                break

    print(f"[match] normalized name matches: {len(name_matches)}")

    # 4. 模糊匹配（未匹配项，低阈值记录供抽检）
    rem_ms = {k: v for k, v in ms_models.items() if k not in matched_ms}
    rem_hf = {k: v for k, v in hf_models.items() if k not in matched_hf}

    # 构建 token -> HF 列表 的倒排索引
    hf_index = defaultdict(list)
    for hf_key, hf in rem_hf.items():
        _, hf_name = parse_id(hf.get("id") or hf.get("modelId") or "")
        _, hf_tokens = _norm_and_tokens(hf_name)
        for tok in hf_tokens:
            hf_index[tok].append((hf_key, hf))

    print(f"[match] fuzzy phase: {len(rem_ms)} MS x {len(rem_hf)} HF (token-indexed)")

    ms_items = list(rem_ms.items())
    n_workers = min(workers, cpu_count() or 1)
    batch_size = max(1, len(ms_items) // n_workers)
    batches = [ms_items[i:i + batch_size] for i in range(0, len(ms_items), batch_size)]

    fuzzy_candidates = []
    if n_workers > 1 and len(ms_items) > 1000:
        with Pool(n_workers) as pool:
            for batch_result in pool.imap_unordered(_fuzzy_worker,
                                                     [(b, hf_index, fuzzy_low_threshold) for b in batches]):
                fuzzy_candidates.extend(batch_result)
    else:
        for batch in batches:
            fuzzy_candidates.extend(_fuzzy_worker((batch, hf_index, fuzzy_low_threshold)))

    # 去重 fuzzy：每个模型只保留最佳匹配
    fuzzy_candidates.sort(key=lambda x: -x[0])
    used_ms2 = set()
    used_hf2 = set()
    fuzzy_matches = []
    for score, ms_key, hf_key, ms, hf in fuzzy_candidates:
        if ms_key in used_ms2 or hf_key in used_hf2:
            continue
        used_ms2.add(ms_key)
        used_hf2.add(hf_key)
        fuzzy_matches.append((ms, hf, "fuzzy"))
        matched_ms.add(ms_key)
        matched_hf.add(hf_key)

    print(f"[match] fuzzy matches: {len(fuzzy_matches)}")

    return exact_matches + alias_matches + name_matches + fuzzy_matches, matched_ms, matched_hf


def build_records(matches, ms_models, hf_models, matched_ms, matched_hf, matched_only=False):
    """把匹配结果组装为最终记录。"""
    records = []
    for ms, hf, method in matches:
        ms_id = ms.get("Id") or ms.get("id") or ""
        hf_id = hf.get("id") or hf.get("modelId") or ""
        ms_dl = ms.get("Downloads") or ms.get("downloads") or 0
        hf_dl = hf.get("downloads") or 0
        ms_lic = extract_license_ms(ms)
        hf_lic = extract_license_hf(hf)

        ratio = None
        try:
            ratio = float(hf_dl) / float(ms_dl) if float(ms_dl) > 0 else None
        except Exception:
            pass

        records.append({
            "label": "",
            "msap_id": ms_id,
            "hf_id": hf_id,
            "platform_presence": "both",
            "match_method": method,
            "has_hf_counterpart": True,
            "hf_gated": bool(hf.get("gated")),
            "downloads_msap": ms_dl,
            "downloads_hf": hf_dl,
            "download_ratio_hf_over_msap": ratio,
            "license_msap": ms_lic,
            "license_hf": hf_lic,
            "license_match": ms_lic == hf_lic if (ms_lic and hf_lic) else None,
            "hf_readme_length": 0,
            "msap_readme_length": 0,
            "tags_hf": hf.get("tags", []),
            "last_modified_hf": hf.get("lastModified", ""),
            "created_at_hf": hf.get("createdAt", ""),
        })

    if not matched_only:
        # 仅魔搭有
        for key, ms in ms_models.items():
            if key in matched_ms:
                continue
            ms_id = ms.get("Id") or ms.get("id") or ""
            records.append({
                "label": "",
                "msap_id": ms_id,
                "hf_id": None,
                "platform_presence": "ms_only",
                "match_method": "",
                "has_hf_counterpart": False,
                "hf_gated": None,
                "downloads_msap": ms.get("Downloads") or ms.get("downloads") or 0,
                "downloads_hf": None,
                "download_ratio_hf_over_msap": None,
                "license_msap": extract_license_ms(ms),
                "license_hf": None,
                "license_match": None,
                "hf_readme_length": 0,
                "msap_readme_length": 0,
                "tags_hf": [],
                "last_modified_hf": "",
                "created_at_hf": "",
            })

        # 仅 HF 有
        for key, hf in hf_models.items():
            if key in matched_hf:
                continue
            hf_id = hf.get("id") or hf.get("modelId") or ""
            records.append({
                "label": "",
                "msap_id": None,
                "hf_id": hf_id,
                "platform_presence": "hf_only",
                "match_method": "",
                "has_hf_counterpart": False,
                "hf_gated": bool(hf.get("gated")),
                "downloads_msap": None,
                "downloads_hf": hf.get("downloads") or 0,
                "download_ratio_hf_over_msap": None,
                "license_msap": None,
                "license_hf": extract_license_hf(hf),
                "license_match": None,
                "hf_readme_length": 0,
                "msap_readme_length": 0,
                "tags_hf": hf.get("tags", []),
                "last_modified_hf": hf.get("lastModified", ""),
                "created_at_hf": hf.get("createdAt", ""),
            })

    return records


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="HF ↔ 魔搭跨平台匹配表更新")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85,
                        help="名称模糊匹配阈值（默认 0.85）")
    parser.add_argument("--workers", type=int, default=4,
                        help="模糊匹配多进程数（默认 4）")
    parser.add_argument("--skip-fuzzy", action="store_true",
                        help="跳过 fuzzy 匹配，只保留 exact/alias/name（大幅提速）")
    parser.add_argument("--matched-only", action="store_true",
                        help="只输出匹配成功的模型对，跳过 ms_only / hf_only（大幅提速）")
    parser.add_argument("--stats-only", action="store_true", help="仅打印统计，不写文件")
    args = parser.parse_args()

    print("[match] 加载魔搭模型...")
    ms_models = load_ms_models(MS_FILE)
    print(f"[match] 魔搭模型: {len(ms_models)}")

    print("[match] 加载 HF 模型...")
    hf_models = load_hf_models(HF_FILE)
    print(f"[match] HF 模型: {len(hf_models)}")

    print("[match] 开始匹配...")
    global matched_ms, matched_hf
    matches, matched_ms, matched_hf = build_match_table(
        ms_models, hf_models,
        fuzzy_threshold=args.fuzzy_threshold,
        workers=args.workers,
    )

    if args.skip_fuzzy:
        # 去掉 fuzzy 匹配结果
        matches = [m for m in matches if m[2] != "fuzzy"]
        matched_ms = set((m[0].get("Id") or m[0].get("id") or "").lower() for m in matches)
        matched_hf = set((m[1].get("id") or m[1].get("modelId") or "").lower() for m in matches)

    print(f"[match] 匹配完成: both={len([m for m in matches if m[2] in ('exact','owner_alias','name')])}, "
          f"fuzzy={len([m for m in matches if m[2]=='fuzzy'])}")
    print(f"[match] 仅魔搭有: {len(ms_models)-len(matched_ms)}, 仅 HF 有: {len(hf_models)-len(matched_hf)}")

    records = build_records(matches, ms_models, hf_models, matched_ms, matched_hf,
                            matched_only=args.matched_only)

    # 统计
    presence_count = {}
    method_count = {}
    for r in records:
        presence_count[r["platform_presence"]] = presence_count.get(r["platform_presence"], 0) + 1
        if r.get("match_method"):
            method_count[r["match_method"]] = method_count.get(r["match_method"], 0) + 1

    print("[match] platform_presence 分布:", presence_count)
    print("[match] match_method 分布:", method_count)

    if args.stats_only:
        return

    # 写 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 写 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {}
            for k in CSV_FIELDS:
                v = r.get(k)
                if isinstance(v, (list, dict, tuple)):
                    v = json.dumps(v, ensure_ascii=False)
                row[k] = v
            writer.writerow(row)

    print(f"[match] 已保存: {OUT_JSON}, {OUT_CSV}")


if __name__ == "__main__":
    main()
