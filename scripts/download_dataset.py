"""Download a pinned dataset into <DEEPTRACE_ROOT>/datasets/<name>/.

  python scripts/download_dataset.py --dataset deepfakeface --list   # sizes only, downloads nothing
  python scripts/download_dataset.py --dataset deepfakeface          # real download (use Colab/Drive)
"""
import argparse
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths
from deeptrace.data.download import dataset_entry, download_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None, help="Override DEEPTRACE_ROOT")
    p.add_argument("--list", action="store_true", help="Show remote file sizes and exit")
    a = p.parse_args()

    cfg = load_config(a.config)
    entry = dataset_entry(cfg, a.dataset)
    paths = get_paths(a.root, create=not a.list)
    download_dataset(a.dataset, entry, paths.datasets, list_only=a.list)


if __name__ == "__main__":
    main()
