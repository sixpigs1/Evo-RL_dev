import pickle
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer


def test_server_passes_requested_device_before_constructing_policy():
    assert "evo1" in SUPPORTED_POLICIES
    server = PolicyServer(PolicyServerConfig())
    specs = RemotePolicyConfig("evo1", "local/checkpoint", {}, 5, device="cpu")
    config = SimpleNamespace(device="cuda:0")
    policy = Mock()
    cls = Mock()
    cls.from_pretrained.return_value = policy
    with (
        patch("lerobot.async_inference.policy_server.PreTrainedConfig.from_pretrained", return_value=config),
        patch("lerobot.async_inference.policy_server.get_policy_class", return_value=cls),
        patch(
            "lerobot.async_inference.policy_server.make_pre_post_processors", return_value=(Mock(), Mock())
        ) as processors,
    ):
        server.SendPolicyInstructions(
            SimpleNamespace(data=pickle.dumps(specs)), SimpleNamespace(peer=lambda: "test")
        )
    assert cls.from_pretrained.call_args.kwargs["config"].device == "cpu"
    policy.eval.assert_called_once()
    assert processors.call_args.kwargs["postprocessor_overrides"]["device_processor"]["device"] == "cpu"


def test_server_chunk_inference_disables_gradients():
    server = PolicyServer(PolicyServerConfig())
    server.device = "cpu"
    server.actions_per_chunk = 3

    def predict(observation):
        assert not torch.is_grad_enabled()
        return torch.ones(1, 5, 12)

    server.policy = SimpleNamespace(config=SimpleNamespace(use_amp=False), predict_action_chunk=predict)
    assert server._get_action_chunk({}).shape == (1, 3, 12)
