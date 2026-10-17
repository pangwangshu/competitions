"""Prevent missing evidence from being reported as action equivalence."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'report' / 'extract_runs.py'
spec = importlib.util.spec_from_file_location('report_extract', SCRIPT)
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class ExtractEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        extract.RUNS = self.root

    def table(self, name, seeds=(1,)):
        with (self.root / (name + '.csv')).open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'score', 'runtime_ms', 'ok'])
            w.writerows((seed, 0, 1, 1) for seed in seeds)

    def diag(self, name, seed=1, complete=True):
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        records = [{'type': 'initial'}, {'type': 'turn', 'action': '-1'}]
        if complete:
            records.append({'type': 'end'})
        (folder / f'{seed}.diag').write_text(''.join('DIAG ' + json.dumps(d) + '\n' for d in records))

    def test_missing_diagnostics_cannot_be_hashed(self):
        self.table('B0')
        with self.assertRaisesRegex(ValueError, 'missing diagnostics'):
            extract.load('B0')

    def test_truncated_diagnostics_rejected(self):
        self.table('B0')
        self.diag('B0', complete=False)
        with self.assertRaisesRegex(ValueError, 'incomplete diagnostics'):
            extract.load('B0')

    def test_duplicate_seed_rejected(self):
        self.table('B0', (1, 1))
        self.diag('B0')
        with self.assertRaisesRegex(ValueError, 'duplicate seed'):
            extract.load('B0')

    def test_failed_run_reported_before_missing_diagnostics(self):
        with (self.root / 'B0.csv').open('w', newline='') as f:
            csv.writer(f).writerows([['seed', 'score', 'runtime_ms', 'ok'], [1, -1, 1, 0]])
        with self.assertRaisesRegex(ValueError, 'seed 1 failed'):
            extract.load('B0')

    def test_valid_trace_has_action_hash(self):
        self.table('B0')
        self.diag('B0')
        rows, _ = extract.load('B0')
        self.assertEqual(rows[1]['action_sha256'], extract.hashlib.sha256(b'-1').hexdigest())

    def test_wrong_run_name_fails_before_output(self):
        self.table('o01c')
        self.table('B0')
        out = self.root / 'output'
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.root), str(out)], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('required O-01C.csv missing', proc.stderr)
        self.assertFalse(out.exists())

    def test_seed_mismatch_cannot_silently_drop_cases(self):
        for name, seeds in [('B0', (1,)), ('O-01C', (1, 2))]:
            self.table(name, seeds)
            for seed in seeds:
                self.diag(name, seed)
        out = self.root / 'output'
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.root), str(out)], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('seed set differs', proc.stderr)
        self.assertFalse(out.exists())
