"""Reproducible offline Evo-RL rounds with explicit task environments and checkpoints."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
cfg = json.loads(Path(a.config).read_text())
if cfg["policy"] not in ("pi05", "evo1"):
    raise ValueError("policy must be pi05 or evo1")
n = cfg.get("gpus", 1)
if n not in (1, 2, 8):
    raise ValueError("gpus must be 1, 2 or 8")
out = Path(cfg["output"])
root = out / "dataset"
repo = cfg["dataset_repo_id"]
steps = cfg.get("steps", 100)
state_path = out / "state.json"
config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
state = (
    json.loads(state_path.read_text()) if state_path.exists() else {"config_hash": config_hash, "stages": {}}
)
if state["config_hash"] != config_hash:
    raise ValueError("Round config changed: use a new output directory")


def save():
    temp = state_path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(state_path)


def run(stage, command, artifact=None):
    if a.dry_run:
        print(json.dumps({"stage": stage, "command": list(map(str, command)), "artifact": str(artifact)}))
        return
    if state["stages"].get(stage, {}).get("status") == "passed":
        if artifact and not Path(artifact).exists():
            raise FileNotFoundError(f"Completed stage artifact missing: {artifact}")
        return
    out.mkdir(parents=True, exist_ok=True)
    state["stages"][stage] = {"status": "running", "command": list(map(str, command)), "started": time.time()}
    save()
    with (out / f"{stage}.log").open("a") as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    artifact_ok = artifact is None or Path(artifact).exists()
    state["stages"][stage].update(
        status="passed" if result.returncode == 0 and artifact_ok else "failed",
        returncode=result.returncode,
        finished=time.time(),
    )
    save()
    result.check_returncode()
    if artifact and not Path(artifact).exists():
        raise FileNotFoundError(artifact)


def command(profile, script, arguments, distributed=True):
    prefix = [ROOT / "tools/musa/run.sh", profile]
    if distributed:
        prefix += [
            "accelerate",
            "launch",
            "--config_file",
            ROOT / f"tools/musa/configs/{n}gpu.yaml",
            "-m",
            f"lerobot.scripts.{script}",
        ]
    else:
        prefix += ["python", ROOT / script]
    return prefix + arguments


def opts(values):
    return [f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in values.items()]


if not a.dry_run:
    out.mkdir(parents=True, exist_ok=True)
    if "prepare" not in state["stages"]:
        if root.exists():
            raise FileExistsError("Untracked dataset copy exists; use a fresh round output")
        # merge via repository implementation, preserving and reindexing metadata.
        sources = [{"repo_id": repo, "root": cfg["dataset_root"]}] + cfg.get("additional_datasets", [])
        state["sources"] = sources
        for item in sources:
            info = Path(item["root"]) / "meta/info.json"
            item["metadata_sha256"] = hashlib.sha256(info.read_bytes()).hexdigest()
        if len(sources) == 1:
            shutil.copytree(sources[0]["root"], root, ignore=shutil.ignore_patterns(".cache"))
        else:
            # This branch runs in the value environment selected by run.sh.
            from lerobot.datasets.dataset_tools import merge_datasets
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            datasets = [LeRobotDataset(x["repo_id"], root=x["root"], video_backend="pyav") for x in sources]
            merge_datasets(datasets, output_repo_id=repo, output_dir=root)
        state["stages"]["prepare"] = {"status": "passed", "replayed_data": cfg.get("replayed_data", False)}
        state["git_commit"] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
            ).stdout.strip()
            or "source archive"
        )
        state["reused_inputs"] = {
            k: cfg[k]
            for k in ("value_checkpoint", "sft_checkpoint", "value_pretrained_checkpoint")
            if cfg.get(k)
        }
        save()
run(
    "check-data",
    command(
        "value",
        "tools/musa/check_dataset.py",
        ["--root", root, "--repo-id", repo, "--require-labels", "--output", out / "data-report.json"]
        + (["--require-acp"] if cfg.get("reuse_annotations") else []),
        False,
    ),
)
common = {
    "dataset.repo_id": repo,
    "dataset.root": root,
    "dataset.video_backend": "pyav",
    "dataset.image_transforms.enable": False,
    "batch_size": cfg.get("batch_size", 1),
    "num_workers": 0,
    "steps": steps,
    "save_freq": steps,
    "log_freq": 1,
    "wandb.enable": False,
    "save_checkpoint": True,
}
value_out = out / "value"
value_args = common | {
    "value.type": "pistar06",
    "value.device": "musa",
    "value.dtype": "bfloat16",
    "value.push_to_hub": False,
    "value.use_gradient_checkpointing": True,
    "value.vision_repo_id": cfg["value_vision"],
    "value.language_repo_id": cfg["value_language"],
    "output_dir": value_out,
}
if cfg.get("value_pretrained_checkpoint"):
    value_args["value.path"] = cfg["value_pretrained_checkpoint"]
    value_args.pop("value.type", None)
value_checkpoint = Path(cfg["value_checkpoint"]) if cfg.get("value_checkpoint") else value_out
if cfg.get("value_checkpoint"):
    if not a.dry_run and not (value_checkpoint / "config.json").exists():
        raise FileNotFoundError(value_checkpoint)
else:
    run(
        "value-train",
        command("value", "lerobot_value_train", opts(value_args)),
        value_out / "checkpoints/last/pretrained_model",
    )
if not cfg.get("reuse_annotations"):
    run(
        "value-infer",
        command(
            "value",
            "lerobot_value_infer",
            opts(
                {
                    "dataset.repo_id": repo,
                    "dataset.root": root,
                    "runtime.device": "musa",
                    "runtime.batch_size": 4,
                    "runtime.num_workers": 0,
                    "inference.checkpoint_path": value_checkpoint,
                    "acp.enable": True,
                    "acp.value_field": "complementary_info.value_musa",
                    "acp.advantage_field": "complementary_info.advantage_musa",
                    "acp.indicator_field": "complementary_info.acp_indicator_musa",
                    "output_dir": out / "annotations",
                }
            ),
        ),
    )
policy = cfg["policy"]
base = common | {"policy.device": "musa", "policy.push_to_hub": False, "eval_freq": 0}
if cfg.get("rename_map"):
    base["rename_map"] = json.dumps(cfg["rename_map"])
if policy == "pi05":
    base |= {
        "policy.path": cfg["policy_model"],
        "policy.dtype": "bfloat16",
        "policy.gradient_checkpointing": True,
        "policy.compile_model": False,
        "policy.use_amp": True,
    }
    if cfg.get("tokenizer_name"):
        base["policy.tokenizer_name"] = cfg["tokenizer_name"]
else:
    base |= {
        "policy.type": "evo1",
        "policy.vlm_model_name": cfg["policy_model"],
        "policy.use_flash_attn": False,
        "policy.training_stage": "stage1",
        "policy.enable_vlm_gradient_checkpointing": True,
    }
sft = out / "sft"
sft_checkpoint = (
    Path(cfg["sft_checkpoint"]) if cfg.get("sft_checkpoint") else sft / "checkpoints/last/pretrained_model"
)
if cfg.get("sft_checkpoint"):
    if not a.dry_run and not (sft_checkpoint / "config.json").exists():
        raise FileNotFoundError(sft_checkpoint)
else:
    run("sft", command(policy, "lerobot_train", opts(base | {"output_dir": sft})), sft_checkpoint)
acp = base | {
    "policy.path": sft_checkpoint,
    "resume_pretrain": True,
    "acp.enable": True,
    "acp.indicator_field": "complementary_info.acp_indicator_musa",
    "acp.indicator_dropout_prob": 0.3,
    "output_dir": out / "acp",
}
acp.pop("policy.type", None)
if policy == "evo1":
    acp["policy.training_stage"] = "stage2"
run("acp-train", command(policy, "lerobot_train", opts(acp)), out / "acp/checkpoints/last/pretrained_model")
run(
    "action-inference",
    command(
        policy,
        "tools/musa/infer_policy.py",
        [
            "--checkpoint",
            out / "acp/checkpoints/last/pretrained_model",
            "--dataset-root",
            root,
            "--repo-id",
            repo,
            "--device",
            cfg.get("inference_device", "musa:0"),
            "--output",
            out / "actions.json",
        ],
        False,
    ),
    out / "actions.json",
)
if not a.dry_run:
    export = {
        "policy_checkpoint": str(out / "acp/checkpoints/last/pretrained_model"),
        "policy": policy,
        "source_round": str(out),
        "next_dataset_repo_id": cfg.get("next_dataset_repo_id", "local/next-round"),
        "robot_execution": "manual; use lerobot-human-inloop-record with the actual robot configuration",
        "replayed_data": cfg.get("replayed_data", False),
    }
    (out / "deployment.json").write_text(json.dumps(export, indent=2))
    print(json.dumps(export, indent=2))
