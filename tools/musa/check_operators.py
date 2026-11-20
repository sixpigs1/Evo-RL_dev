"""Exercise Pi0.5 sampling/embedding and shared attention with CPU references."""

import json

import torch
import torch_musa  # noqa: F401

from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding, sample_beta

d = torch.device("musa:4")
time = torch.tensor([0.1, 0.5, 0.9])
ref = create_sinusoidal_pos_embedding(time, 256, 0.004, 4.0, torch.device("cpu"))
result = create_sinusoidal_pos_embedding(time.to(d), 256, 0.004, 4.0, d)
assert torch.allclose(ref.float(), result.float().cpu(), atol=0.0005, rtol=0.0005)
samples = sample_beta(1.5, 1.0, 1024, d)
assert samples.device == d and torch.isfinite(samples).all() and ((samples >= 0) & (samples <= 1)).all()
q = torch.randn(2, 4, 16, 32, device=d, requires_grad=True)
mask = torch.ones(16, 16, device=d, dtype=torch.bool).tril()
output = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
output.square().mean().backward()
assert torch.isfinite(q.grad).all()
print(
    json.dumps(
        {
            "status": "passed",
            "embedding_max_abs_error": (ref.float() - result.float().cpu()).abs().max().item(),
            "beta_mean": samples.mean().item(),
        }
    )
)
