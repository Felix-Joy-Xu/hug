#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 对照组采集
=====================
对魔搭 18 个模型卡片样本，抓同名 HuggingFace 模型卡片 + 元数据
回答: 中文→英文 / 国产→国际 的镜像关系

输出:
  modelscope_output/hf_cards/    18 个 README.md
  modelscope_output/hf_metadata.json + .csv
  modelscope_output/cross_platform_match.json    魔搭↔HF 同名对照表
"""
import os, json, time, csv, requests

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "modelscope_output")
HF_CARDS_DIR = os.path.join(OUTPUT_DIR, "hf_cards")
os.makedirs(HF_CARDS_DIR, exist_ok=True)

# 魔搭 18 个样本 → HF 对应名（多数直接同名/大小写不同）
SAMPLES = [
    # (msap_id, hf_id, label)
    ("qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "通义千问2.5"),
    ("qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "通义千问2.5-72B"),
    ("Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B", "通义千问3旗舰MoE"),
    ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3", "DeepSeek-V3"),
    ("deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
    ("ZhipuAI/glm-4-9b", "THUDM/glm-4-9b", "智谱GLM-4"),
    ("ZhipuAI/glm-4-9b-chat", "THUDM/glm-4-9b-chat", "智谱GLM-4对话"),
    ("baichuan-inc/Baichuan2-13B-Chat", "baichuan-inc/Baichuan2-13B-Chat", "百川2"),
    ("01ai/Yi-1.5-34B-Chat", "01-ai/Yi-1.5-34B-Chat", "零一万物Yi"),
    ("internlm/internlm2_5-7b-chat", "internlm/internlm2_5-7b-chat", "InternLM2.5"),
    ("LLM-Research/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B"),
    ("LLM-Research/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),
    ("AI-ModelScope/gemma-2-2b-it", "google/gemma-2-2b-it", "Google Gemma-2"),
    ("mlx-community/Phi-3.5-mini-instruct-8bit", "microsoft/Phi-3-mini-128k-instruct", "微软Phi-3"),
    ("qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL多模态"),
    ("AI-ModelScope/InternVL-Chat-V1-5", "OpenGVLab/InternVL-Chat-V1-5", "InternVL多模态"),
    ("iic/nlp_structbert_word-segmentation_chinese-base", None, "达摩院中文分词(没有HF镜像)"),
    ("damo/nlp_gpt3_text-generation_1.3B", None, "达摩院GPT3(没有HF镜像)"),
    ("XiaomiMiMo/XiaomiMiMo-VL-7B-RL-2508", "XiaomiMiMo/XiaomiMiMo-VL-7B-RL-2508", "小米MiMo-VL"),
    ("AI-ModelScope/Hunyuan-7B-Instruct", "tencent/Hunyuan-7B-Instruct", "腾讯混元7B"),
]

HF_BASE = os.environ.get("HF_BASE", "https://hf-mirror.com")
HF_API = HF_BASE + "/api/models/{}"
HF_RAW = HF_BASE + "/{}/raw/main/README.md"
HF_RAW_FALLBACK = HF_BASE + "/{}/raw/master/README.md"

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain"})

matches = []
all_meta = []

for msap_id, hf_id, label in SAMPLES:
    print(f"\n--- {label} ---")
    print(f"  ModelScope: {msap_id}")

    if hf_id is None:
        print(f"  HF: 无对照")
        matches.append({
            "msap_id": msap_id, "hf_id": None, "label": label,
            "has_hf_counterpart": False,
            "downloads_ms": None, "downloads_hf": None,
            "license_ms": None, "license_hf": None,
        })
        continue

    print(f"  HF: {hf_id}")

    # 1. HF 模型元数据
    try:
        r = s.get(HF_API.format(hf_id), timeout=15)
        if r.status_code == 200:
            hf_meta = r.json()
        elif r.status_code == 401:
            print(f"    [HF gated] 模型受控访问")
            hf_meta = {"gated": True}
        else:
            print(f"    [HF {r.status_code}] 未找到")
            hf_meta = None
    except Exception as e:
        print(f"    [ERR] {str(e)[:50]}")
        hf_meta = None

    # 2. HF README 全文
    readme_content = None
    readme_url = None
    if hf_meta and not hf_meta.get("gated"):
        for url_pattern in [HF_RAW, HF_RAW_FALLBACK]:
            try:
                r = s.get(url_pattern.format(hf_id), timeout=15)
                if r.status_code == 200 and len(r.text) > 50:
                    readme_content = r.text
                    readme_url = url_pattern.format(hf_id)
                    break
            except:
                continue

    # 保存 README
    if readme_content:
        safe_name = hf_id.replace("/", "_")
        with open(os.path.join(HF_CARDS_DIR, f"{safe_name}_README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"    README: {len(readme_content)} chars")

    # 3. 加载魔搭侧元数据
    with open(os.path.join(OUTPUT_DIR, "models_all.json"), "r", encoding="utf-8") as f:
        ms_models = json.load(f)
    msap_meta = None
    for m in ms_models:
        mid = m.get("Id") or m.get("id") or ""
        if mid.lower() == msap_id.lower():
            msap_meta = m
            break
    if not msap_meta:
        # 索引中也有（索引文件可能不存在，如云端环境）
        idx_path = os.path.join(OUTPUT_DIR, "model_cards_index.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                cards_idx = json.load(f)
            for c in cards_idx:
                if c["model_id"].lower() == msap_id.lower():
                    msap_meta = c
                    break

    # 4. 组装对比记录
    msap_dl = (msap_meta or {}).get("Downloads") or (msap_meta or {}).get("downloads") or 0
    msap_lic = (msap_meta or {}).get("License") or (msap_meta or {}).get("license", "")
    hf_dl = (hf_meta or {}).get("downloads", 0) if hf_meta else 0
    hf_lic = ((hf_meta or {}).get("cardData") or {}).get("license", "") if hf_meta else ""
    hf_tags = (hf_meta or {}).get("tags", []) if hf_meta else []
    hf_likes = (hf_meta or {}).get("likes", 0) if hf_meta else 0
    hf_modified = (hf_meta or {}).get("lastModified", "") if hf_meta else ""

    record = {
        "label": label,
        "msap_id": msap_id,
        "hf_id": hf_id,
        "has_hf_counterpart": hf_meta is not None,
        "hf_gated": bool(hf_meta and hf_meta.get("gated")),
        "downloads_msap": msap_dl,
        "downloads_hf": hf_dl,
        "download_ratio_hf_over_msap": (hf_dl / msap_dl) if msap_dl else None,
        "license_msap": msap_lic,
        "license_hf": hf_lic,
        "license_match": (msap_lic.lower() == (hf_lic or "").lower()) if hf_lic else None,
        "hf_readme_length": len(readme_content or ""),
        "msap_readme_length": 0,  # 之后填充
        "tags_hf": (hf_meta or {}).get("tags", []) if hf_meta else [],
        "last_modified_hf": (hf_meta or {}).get("lastModified", "") if hf_meta else "",
        "created_at_hf": (hf_meta or {}).get("createdAt", "") if hf_meta else "",
    }
    # 用魔搭 README 长度
    msap_card_path = os.path.join(OUTPUT_DIR, "model_cards", msap_id.replace("/", "_") + "_README.md")
    if os.path.exists(msap_card_path):
        with open(msap_card_path, "r", encoding="utf-8") as f:
            record["msap_readme_length"] = len(f.read())

    matches.append(record)
    all_meta.append({
        "platform": "HF",
        "id": f"{hf_id}/README",
        "label": label,
        "readme_length": len(readme_content or ""),
        "has_license_section": (readme_content or "").lower().find("license") > -1 if readme_content else False,
        "has_disclaimer": any(k in (readme_content or "").lower() for k in ["disclaimer", "免责", "声明", "使用限制"]),
        "has_chinese_narrative": any(k in (readme_content or "") for k in ["国产", "自主", "中国", "开源"]),
        "has_training_data": ("training data" in (readme_content or "").lower() or "训练数据" in (readme_content or "")) if readme_content else False,
    })

    print(f"    HF dl={hf_dl}, MS dl={msap_dl}, ratio hf/ms={record['download_ratio_hf_over_msap']}")
    print(f"    HF lic={hf_lic}, MS lic={msap_lic}, match={record['license_match']}")

    time.sleep(0.5)

# 保存
with open(os.path.join(OUTPUT_DIR, "cross_platform_match.json"), "w", encoding="utf-8") as f:
    json.dump(matches, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTPUT_DIR, "hf_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(all_meta, f, ensure_ascii=False, indent=2)

# CSV
with open(os.path.join(OUTPUT_DIR, "cross_platform_match.csv"), "w", newline="", encoding="utf-8-sig") as f:
    fields = ["label", "msap_id", "hf_id", "has_hf_counterpart", "hf_gated",
              "downloads_msap", "downloads_hf", "download_ratio_hf_over_msap",
              "license_msap", "license_hf", "license_match",
              "hf_readme_length", "msap_readme_length",
              "last_modified_hf", "created_at_hf"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for m in matches:
        w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list)
                        else (v.isoformat() if hasattr(v, "isoformat") else v))
                    for k, v in m.items()})

print()
print("="*60)
print("HuggingFace 对照组采集完成")
print("="*60)
print(f"  对照模型对数: {len(matches)}")
print(f"  HF README 拉到: {sum(1 for m in all_meta if m['readme_length'] > 0)}")
print(f"  HF 卡片目录: {HF_CARDS_DIR}/")
print(f"  对照表: cross_platform_match.json/.csv")

# 汇总
have_hf = [m for m in matches if m["has_hf_counterpart"] and not m["hf_gated"]]
no_hf = [m for m in matches if not m["has_hf_counterpart"] or m["hf_gated"]]
license_match = [m for m in have_hf if m["license_match"]]
license_diff = [m for m in have_hf if m["license_match"] is False]

print(f"\n  有 HF 对照且可访问: {len(have_hf)}/{len(matches)}")
print(f"  无 HF/受控访问:    {len(no_hf)}/{len(matches)}")
print(f"  许可证一致:        {len(license_match)}/{len(have_hf)}")
print(f"  许可证不一致:      {len(license_diff)}/{len(have_hf)}")
if license_diff:
    for m in license_diff:
        print(f"    - {m['label']}: MS={m['license_msap']} | HF={m['license_hf']}")