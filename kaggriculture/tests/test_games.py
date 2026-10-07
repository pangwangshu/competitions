"""End-to-end: full 720-turn games through the real engine (about 5 s each)."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("kaggle_environments")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAY = os.path.join(ROOT, "scripts", "play.py")


def play(*args):
    out = subprocess.run([sys.executable, PLAY, "--json", *args], capture_output=True,
                         text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("seat", [0, 1])
def test_beats_the_starter_bot_from_either_seat(seat):
    result = play("--opponent", "starter", "--seed", "11", "--seat", str(seat))
    assert result["status"] == ["DONE", "DONE"]
    assert result["margin"] > 20000


def test_self_play_in_one_process():
    # Both seats load the same package in one process; state is per player.
    result = play("--opponent", "self", "--seed", "12")
    assert result["status"] == ["DONE", "DONE"]
    assert min(result["money"]) > 30000


def test_games_are_deterministic():
    a = play("--opponent", "starter", "--seed", "13")
    b = play("--opponent", "starter", "--seed", "13")
    assert a == b


def test_submission_archive_runs_from_a_clean_directory(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import build_submission

    path = build_submission.build(str(tmp_path / "submission.tar.gz"))
    money = build_submission.verify(path)
    assert float(money[0]) > float(money[1])
