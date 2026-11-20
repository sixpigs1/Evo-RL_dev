"""Install isolated task dependencies while retaining the vendor torch stack."""

import argparse
import fcntl
import importlib.metadata as md
import json
import os
import subprocess
import venv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("profile", choices=["pi05", "evo1", "value"])
p.add_argument("--storage", default="/workspace/storage")
p.add_argument("--pi-transformers-revision")
p.add_argument("--lock-file", type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
storage = Path(a.storage)
env = storage / "environments" / a.profile
# Vendor packages are deliberately inherited, but task dependencies shadow them.
env.mkdir(parents=True, exist_ok=True)
lock = (env / ".install.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if not (env / "bin/python").exists():
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(env)
cfg_path = env / "pyvenv.cfg"
cfg_path.write_text(
    cfg_path.read_text().replace(
        "include-system-site-packages = false", "include-system-site-packages = true"
    )
)
python = str(env / "bin/python")
versions = {n: md.version(n) for n in ("torch", "torch_musa", "torchvision")}
constraints = env / "vendor-constraints.txt"
constraints.write_text(
    "".join(f"{k}=={v}\n" for k, v in versions.items()) + (a.lock_file.read_text() if a.lock_file else "")
)
os.environ["PIP_CONSTRAINT"] = str(constraints)
os.environ["PIP_CACHE_DIR"] = str(storage / "cache/pip")


def pip(*args):
    subprocess.run([python, "-m", "pip", *args], check=True)


pip("install", "tomli", "wheel")
import tomli  # noqa: E402

project = tomli.loads((root / "pyproject.toml").read_text())["project"]
requirements = project["dependencies"] + [
    "numpy==1.26.4",
    "tokenizers>=0.21,<0.23",
    "opencv-python-headless==4.11.0.86",
    "pytest==8.4.2",
    "pytest-timeout",
    "scipy==1.14.1",
    "sentencepiece",
    "timm==1.0.22",
    "num2words",
    "ruff",
]
if a.profile == "pi05":
    if not a.pi_transformers_revision or len(a.pi_transformers_revision) != 40:
        raise ValueError("Pi0.5 requires a pinned 40-character Transformers commit")
    requirements += [
        f"transformers @ https://github.com/huggingface/transformers/archive/{a.pi_transformers_revision}.zip"
    ]
else:
    requirements += ["transformers==4.51.3" if a.profile == "evo1" else "transformers==4.57.1"]
pip("install", *requirements)
pip("install", "--no-deps", "-e", str(root))
# Refresh source metadata explicitly: old copied egg-info may shadow the editable wheel.
subprocess.run([python, str(root / "setup.py"), "egg_info", "--egg-base", "src"], cwd=root, check=True)
actual = json.loads(
    subprocess.check_output(
        [
            python,
            "-c",
            'import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ["torch","torch_musa","torchvision"]}))',
        ]
    )
)
assert actual == versions, (actual, versions)
# The inherited image contains unrelated serving packages; preserve the complete check output.
check = subprocess.run([python, "-m", "pip", "check"], capture_output=True, text=True)
(env / "pip-check.txt").write_text(check.stdout + check.stderr)
(env / "requirements.freeze.txt").write_bytes(subprocess.check_output([python, "-m", "pip", "freeze"]))
(env / "manifest.json").write_text(
    json.dumps(
        {
            "vendor": actual,
            "profile": a.profile,
            "pip_check_returncode": check.returncode,
            "pi_transformers_revision": a.pi_transformers_revision,
        },
        indent=2,
    )
)
subprocess.run(
    [
        python,
        "-c",
        'import torch,torch_musa,torchvision,transformers,accelerate; from lerobot.utils.devices import require_musa; require_musa("musa")',
    ],
    check=True,
)

subprocess.run([python, str(root / "tools/musa/validate_environment.py")], check=True)
