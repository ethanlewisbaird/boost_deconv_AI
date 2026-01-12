import os
import sys
import re
from pathlib import Path

def sanitize_path(path: str) -> str:
    return path.strip().replace("`", "").replace(" ", "")

def write_file(base_dir: Path, file_path: str, content: str):
    full_path = base_dir / sanitize_path(file_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Wrote: {full_path}")

def extract_code_blocks(md_text: str):
    pattern = re.compile(
        r"```(?P<lang>[a-zA-Z0-9_+-]*)[^\n]*?(path=(?P<path>[^\s]+))?\s*\n(?P<code>.*?)```",
        re.DOTALL
    )
    for m in pattern.finditer(md_text):
        yield {
            "lang": m.group("lang") or "",
            "path": m.group("path"),
            "code": m.group("code")
        }

def reconstruct(md_file, out_dir):
    md_text = Path(md_file).read_text(encoding="utf-8")
    blocks = list(extract_code_blocks(md_text))
    if not blocks:
        print("No code blocks found.")
        sys.exit(1)

    print(f"Found {len(blocks)} blocks.")
    for b in blocks:
        if not b["path"]:
            print("[SKIP] No 'path=' defined for block.")
            continue
        write_file(Path(out_dir), b["path"], b["code"])

    print("Extraction complete!")

if __name__ == "__main__":
    md_file = sys.argv[1]
    out_dir = sys.argv[2]
    reconstruct(md_file, out_dir)
