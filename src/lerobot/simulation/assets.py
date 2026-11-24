"""Fetch and verify the pinned SO-101 and PiPER asset trees."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .menagerie_lock import FILES, REVISION

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
MODEL_FILES = {"so101": "robotstudio_so101/so101.xml", "piper": "agilex_piper/piper.xml"}


def asset_root() -> Path:
    return Path(os.environ.get("LEROBOT_MENAGERIE_ROOT", Path.home() / ".cache/lerobot/menagerie" / REVISION))


def verify_assets(root: Path, model: str | None = None) -> None:
    prefix = f"{Path(MODEL_FILES[model]).parent}/" if model else ""
    for name, expected in FILES.items():
        if not name.startswith(prefix):
            continue
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Invalid or missing Menagerie asset {path}. Run python -m lerobot.simulation.assets."
            )


def prefetch(root: Path) -> None:
    if root.exists():
        verify_assets(root)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        clone = Path(temporary) / "repo"
        subprocess.run(["git", "init", "--quiet", str(clone)], check=True)
        git = ["git", "-C", str(clone)]
        subprocess.run([*git, "remote", "add", "origin", REPOSITORY], check=True)
        subprocess.run([*git, "config", "remote.origin.promisor", "true"], check=True)
        subprocess.run([*git, "config", "remote.origin.partialclonefilter", "blob:none"], check=True)
        subprocess.run([*git, "sparse-checkout", "set", "agilex_piper", "robotstudio_so101"], check=True)
        subprocess.run([*git, "fetch", "--depth=1", "--filter=blob:none", "origin", REVISION], check=True)
        subprocess.run([*git, "checkout", "--detach", "FETCH_HEAD"], check=True)
        verify_assets(clone)
        stage = Path(temporary) / "assets"
        for name in FILES:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(clone / name, target)
        try:
            stage.rename(root)
        except OSError:
            if not root.is_dir():
                raise
            verify_assets(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=asset_root())
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_assets(args.root)
    else:
        prefetch(args.root)
    print(f"Menagerie {REVISION}: {args.root}")


if __name__ == "__main__":
    main()
