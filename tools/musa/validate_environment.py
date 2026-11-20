"""Check the training dependency closure independently of unrelated serving extras."""

import importlib.metadata as md
import json
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

vendor = {"torch": "2.9.0", "torch-musa": "2.9.0", "torchvision": "0.22.1+6b25dcc"}
for name, version in vendor.items():
    assert md.version(name).split("+")[0] == version.split("+")[0], (name, md.version(name))
queue, visited, errors = (
    [
        "lerobot",
        "transformers",
        "accelerate",
        "numpy",
        "grpcio",
        "protobuf",
        "pytest",
        "pytest-timeout",
        "ruff",
        "scipy",
        "timm",
        "sentencepiece",
        "num2words",
    ],
    set(),
    [],
)
while queue:
    name = canonicalize_name(queue.pop())
    if name in visited:
        continue
    visited.add(name)
    # Vendor wheels form one tested ABI unit, not the upstream CUDA wheel matrix.
    if name in vendor:
        continue
    for text in md.requires(name) or []:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        dependency = canonicalize_name(requirement.name)
        try:
            version = md.version(requirement.name)
        except md.PackageNotFoundError:
            errors.append(f"{name}: missing {requirement}")
            continue
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            errors.append(f"{name}: requires {requirement}; installed {version}")
        queue.append(dependency)
report = {
    "status": "passed" if not errors else "failed",
    "python": sys.version,
    "environment": sys.prefix,
    "checked_packages": sorted(visited),
    "errors": errors,
    "vendor_versions": {k: md.version(k) for k in vendor},
}
Path(sys.prefix, "dependency-validation.json").write_text(json.dumps(report, indent=2))
Path(sys.prefix, "requirements.task-lock.txt").write_text(
    "".join(
        f"{name}=={md.version(name)}\n"
        for name in sorted(visited)
        if name not in vendor and name != "lerobot"
    )
)
print(json.dumps(report, indent=2))
if errors:
    raise SystemExit(1)
