"""用 Playwright 补 Terms 和 Community Guidelines（JS 渲染）"""
import os
from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(__file__), "modelscope_output", "hf_governance")

TARGETS = [
    ("terms_of_service", "https://hf-mirror.com/terms", "HF Terms of Service"),
    ("community_guidelines", "https://hf-mirror.com/community-guidelines", "HF Community Guidelines"),
    ("acceptable_use", "https://hf-mirror.com/content-policy", "HF Acceptable Use Policy"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for name, url, label in TARGETS:
        print(f"\n--- {label} ---")
        context = browser.new_context(locale="en-US", viewport={"width": 1280, "height": 900})
        page = context.new_page()
        try:
            page.goto(url, wait_until="load", timeout=30000)
        except:
            pass
        page.wait_for_timeout(5000)
        text = page.evaluate("() => document.body?.innerText || ''")
        if len(text) < 200:
            text = page.content()
            # 临时降级
        print(f"  text len: {len(text)}")
        if len(text) > 200:
            # 清理导航行
            nav_kw = ["Sign in", "Sign up", "Models", "Datasets", "Spaces", "Posts", "Docs", "Enterprise"]
            lines = [l.strip() for l in text.split("\n") if l.strip() and not any(k in l for k in nav_kw)]
            cleaned = "\n".join(lines[:500])
            path = os.path.join(OUT, f"hf_{name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {label}\n# Source: {url}\n# Length: {len(cleaned)} chars\n{'='*80}\n\n{cleaned}")
            print(f"  -> saved ({len(cleaned)} chars)")
            # 预览
            for line in lines[:5]:
                print(f"    | {line[:80]}")
        context.close()
    browser.close()

# 汇总
import os
print(f"\n{'='*60}")
print(f"HF 治理文本最终汇总")
print(f"{'='*60}")
for fn in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, fn))
    print(f"  {fn:40s} {sz:>7,} bytes")