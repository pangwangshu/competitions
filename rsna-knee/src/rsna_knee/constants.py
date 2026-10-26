"""Shared constants for the RSNA knee competition."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
FILELIST_DIR = DATA_DIR / "filelist_pages"
REPORTS_DIR = PROJECT_ROOT / "report"

COMPETITION = "rsna-knee-abnormality-detection"

# 12 abnormality labels, in submission column order.
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PLANES = ["Sagittal", "Coronal", "Axial"]
