import shutil
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from src.common.utils import RESULTS_DIR, input_dirs, work_dir
 
copied = 0
for d in input_dirs():
    for res in d.rglob("results"):
        if not (res.is_dir() and res.parent.name == "repo"):
            continue
        for f in res.rglob("*"):
            rel = f.relative_to(res)
            if not f.is_file() or any("smoke" in part for part in rel.parts):
                continue
            dst = RESULTS_DIR / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                copied += 1
print(f"copied {copied} files from inputs")
 
zip_base = work_dir().parent / "results_bundle"
shutil.make_archive(str(zip_base), "zip", root_dir=RESULTS_DIR.parent, base_dir="results")
print(f"wrote {zip_base}.zip")
for p in sorted(RESULTS_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RESULTS_DIR.parent)}  ({p.stat().st_size / 1024:.0f} KB)")