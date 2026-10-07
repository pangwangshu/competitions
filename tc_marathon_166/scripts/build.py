"""Bundle the src/hextiles package + src/main.py into a single dist/HexTiles.py
for submission (Topcoder requires one flat source file per submission).
"""
import pathlib
import re
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DIST = ROOT / "dist"

# dependency order: later modules may use names defined by earlier ones
MODULES = ["geometry", "model", "io_utils", "incremental", "solver"]

LOCAL_IMPORT_RE = re.compile(r"^\s*from hextiles(\.\w+)? import .*$")


def strip_local_imports(text):
    return "\n".join(
        line for line in text.splitlines() if not LOCAL_IMPORT_RE.match(line)
    )


def main():
    DIST.mkdir(exist_ok=True)

    parts = []
    for mod in MODULES:
        text = (SRC / "hextiles" / f"{mod}.py").read_text()
        parts.append(f"# --- hextiles/{mod}.py ---")
        parts.append(strip_local_imports(text).strip())

    main_text = (SRC / "main.py").read_text()
    parts.append("# --- main.py ---")
    parts.append(strip_local_imports(main_text).strip())

    bundled = "\n\n\n".join(parts) + "\n"

    out_py = DIST / "HexTiles.py"
    out_py.write_text(bundled)

    out_zip = DIST / "HexTiles.py.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_py, arcname="HexTiles.py")

    print(f"wrote {out_py} ({len(bundled)} bytes)")
    print(f"wrote {out_zip}")


if __name__ == "__main__":
    main()
