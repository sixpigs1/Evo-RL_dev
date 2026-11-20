from unittest.mock import patch

import pytest
import torch

from lerobot.utils.devices import require_musa, validate_runtime_device
from lerobot.utils.utils import is_amp_available


def test_explicit_musa_fails_without_runtime():
    with (
        patch("lerobot.utils.devices.musa_available", return_value=False),
        pytest.raises(RuntimeError, match="MUSA requested"),
    ):
        require_musa("musa:0")


def test_musa_runtime_cannot_silently_use_cpu():
    with (
        patch("lerobot.utils.devices.require_musa"),
        pytest.raises(RuntimeError, match="Accelerate selected"),
    ):
        validate_runtime_device("musa", torch.device("cpu"))


def test_indexed_amp_devices():
    assert is_amp_available("musa:3")
    assert is_amp_available("cuda:1")
    assert not is_amp_available("mps")


def test_explicit_processor_device_roundtrip():
    from lerobot.processor.device_processor import DeviceProcessorStep

    step = DeviceProcessorStep("cpu")
    assert step.get_config()["device"] == "cpu"
    x = torch.tensor([1.0])
    assert step._process_tensor(x).device.type == "cpu"


def test_cpu_rng_roundtrip():
    from lerobot.utils.random_utils import deserialize_torch_rng_state, serialize_torch_rng_state

    state = serialize_torch_rng_state()
    first = torch.rand(8)
    deserialize_torch_rng_state(state)
    assert torch.equal(first, torch.rand(8))


def test_explicit_cpu_value_dtype_does_not_depend_on_gpu_presence():
    from lerobot.values.pistar06.modeling_pistar06 import _resolve_load_dtype

    with patch("lerobot.values.pistar06.modeling_pistar06.torch.cuda.is_available", return_value=True):
        assert _resolve_load_dtype("bfloat16", "cpu") == torch.float32
        assert _resolve_load_dtype("bfloat16", "musa:1") == torch.bfloat16


def test_musa_safetensors_deserializes_on_cpu():
    from lerobot.policies.pretrained import PreTrainedPolicy

    model = torch.nn.Linear(2, 2)
    with patch("lerobot.policies.pretrained.load_model_as_safetensor", return_value=([], [])) as loader:
        result = PreTrainedPolicy._load_as_safetensor(model, "checkpoint.safetensors", "musa:3", True)
    assert result is model
    assert loader.call_args.kwargs["device"] == "cpu"


def test_global_batch_metrics_do_not_multiply_world_size_twice():
    from types import SimpleNamespace

    from lerobot.utils.logging_utils import MetricsTracker

    tracker = MetricsTracker(8, 100, 2, {}, initial_step=2, accelerator=SimpleNamespace(num_processes=8))
    tracker.step()
    assert tracker.samples == 24


def test_non_reentrant_vision_checkpoint_keeps_parameter_gradients():
    from types import SimpleNamespace

    from lerobot.policies.evo1.checkpointing import install_non_reentrant_vision_checkpoint

    encoder = torch.nn.Module()
    encoder.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
    encoder.gradient_checkpointing = True
    encoder.config = SimpleNamespace(output_hidden_states=False, use_return_dict=True)
    install_non_reentrant_vision_checkpoint(encoder)
    encoder(torch.randn(2, 3, 4)).last_hidden_state.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_seeded_context_restores_state_on_exception():
    from lerobot.utils.random_utils import seeded_context

    torch.manual_seed(7)
    before = torch.get_rng_state()
    with pytest.raises(ValueError), seeded_context(99):
        torch.rand(5)
        raise ValueError("interrupted")
    assert torch.equal(before, torch.get_rng_state())


def test_disabled_checkpoint_action_steps_are_explicit_noops():
    from lerobot.processor.compatibility_processor import DisabledRelativeActionsProcessorStep

    transition = {"action": torch.ones(2)}
    assert DisabledRelativeActionsProcessorStep()(transition) is transition
    with pytest.raises(ValueError, match="only disabled"):
        DisabledRelativeActionsProcessorStep(enabled=True)
