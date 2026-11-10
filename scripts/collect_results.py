"""Gather small result and split files from attached notebook outputs and zip them.

  python scripts/collect_results.py
Copies every file under <input>/**/repo/results/ and <input>/**/repo/splits/ (skipping smoke
tests) into this repo, without overwriting, then writes /kaggle/working/results_bundle.zip
containing results/ and splits/. Download it, unzip, and drag both folders into github.dev.
"""
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.utils import REPO_ROOT, input_dirs, work_dir

copied = 0
for d in input_dirs():
    for name in ("results", "splits"):
        for src in d.rglob(name):
            if not (src.is_dir() and src.parent.name == "repo"):
                continue
            for f in src.rglob("*"):
                rel = f.relative_to(src)
                if not f.is_file() or any("smoke" in part for part in rel.parts):
                    continue
                dst = REPO_ROOT / name / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    copied += 1
print(f"copied {copied} files from inputs")

zip_path = work_dir().parent / "results_bundle.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("results", "splits"):
        for p in sorted((REPO_ROOT / name).rglob("*")):
            if p.is_file() and "smoke" not in str(p.relative_to(REPO_ROOT)):
                z.write(p, p.relative_to(REPO_ROOT))
print(f"wrote {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MB)")