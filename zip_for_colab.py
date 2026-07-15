"""Create a minimal zip for Colab upload — code only, no models or raw data.

Models and raw data are downloaded on Colab directly (much faster than uploading).
Output: DeepProjectHate_colab.zip (~5 MB)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "DeepProjectHate_colab.zip"

INCLUDE_DIRS = ["src", "scripts", "notebooks", "data/processed", "report/figures"]
INCLUDE_FILES = [
    "README.md",
    "CLAUDE.md",
    "requirements.txt",
]
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".cursor",
    "archive",
}


def should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return False
    if path.suffix in (".pyc", ".pyo"):
        return False
    return True


def main() -> None:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for d in INCLUDE_DIRS:
            dir_path = ROOT / d
            if not dir_path.exists():
                print(f"  skip (missing): {d}/")
                continue
            for f in sorted(dir_path.rglob("*")):
                if f.is_file() and should_include(f.relative_to(ROOT)):
                    arcname = f.relative_to(ROOT).as_posix()
                    zf.write(f, arcname)

        for f in INCLUDE_FILES:
            fp = ROOT / f
            if fp.exists():
                zf.write(fp, f)

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"Created {OUT.name}: {size_mb:.1f} MB")
    print(f"  at: {OUT}")

    print("\nContents:")
    with zipfile.ZipFile(OUT) as zf:
        for info in zf.infolist():
            kb = info.file_size / 1024
            print(f"  {kb:8.1f} KB  {info.filename}")


if __name__ == "__main__":
    main()
