"""看 hf-mirror 是否返回 license"""
import requests
s = requests.Session()
r = s.get("https://hf-mirror.com/api/models/Qwen/Qwen2.5-7B-Instruct", timeout=15)
d = r.json()
print("keys:", list(d.keys())[:20])
for k in ["downloads", "likes", "license", "tags", "lastModified", "siblings"]:
    v = d.get(k)
    if isinstance(v, list):
        print(f"  {k}: list({len(v)}) {v[:3]}")
    else:
        print(f"  {k}: {v}")