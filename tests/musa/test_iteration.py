import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("policy", ["pi05", "evo1"])
def test_round_dry_run_uses_consistent_annotated_dataset(tmp_path, policy):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / f"tools/musa/configs/round-{policy}.json").read_text())
    config["output"] = str(tmp_path / "not-created")
    config_path = tmp_path / "round.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(root / "tools/musa/iteration.py"), "--config", str(config_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [json.loads(x) for x in result.stdout.splitlines()]
    assert [x["stage"] for x in commands] == [
        "check-data",
        "value-train",
        "value-infer",
        "sft",
        "acp-train",
        "action-inference",
    ]
    acp = next(x for x in commands if x["stage"] == "acp-train")["command"]
    assert "--acp.enable=true" in acp
    assert "--acp.indicator_field=complementary_info.acp_indicator_musa" in acp
    if policy == "evo1":
        assert "--policy.training_stage=stage2" in acp
        assert "--resume_pretrain=true" in acp
    assert not Path(config["output"]).exists()
