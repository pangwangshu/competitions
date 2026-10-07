"""Build the Kaggle submission archive and verify it runs from a clean directory.

    python scripts/build_submission.py            # writes dist/submission.tar.gz
    kaggle competitions submit kaggriculture -f dist/submission.tar.gz -m "message"

Verification extracts the archive into an empty temporary directory and plays a
full game from there, which is how the Kaggle runner sees it: main.py at the
root, executed without __file__, from an unrelated working directory.
"""

import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dist", "submission.tar.gz")


def build(path=OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def keep(info):
        return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info

    with tarfile.open(path, "w:gz") as tar:
        tar.add(os.path.join(ROOT, "main.py"), arcname="main.py")
        tar.add(os.path.join(ROOT, "farm_agent"), arcname="farm_agent", filter=keep)
    return path


def verify(path):
    work = tempfile.mkdtemp()
    with tarfile.open(path) as tar:
        try:
            tar.extractall(work, filter="data")
        except TypeError:  # Python < 3.12
            tar.extractall(work)
    code = (
        "from kaggle_environments import make\n"
        "env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1})\n"
        f"env.run([{os.path.join(work, 'main.py')!r}, 'starter'])\n"
        "final = env.steps[-1]\n"
        "assert [s.status for s in final] == ['DONE', 'DONE'], [s.status for s in final]\n"
        "print(final[0].reward, final[1].reward)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=tempfile.mkdtemp(),
                         capture_output=True, text=True, check=True)
    return out.stdout.split()[-2:]


def main():
    path = build()
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with tarfile.open(path) as tar:
        n = len(tar.getnames())
    money = verify(path)
    print(f"{os.path.relpath(path, ROOT)}: {n} entries, sha256 {digest[:16]}...")
    print(f"verified from a clean directory: agent ${float(money[0]):,.0f} vs starter ${float(money[1]):,.0f}")


if __name__ == "__main__":
    main()
