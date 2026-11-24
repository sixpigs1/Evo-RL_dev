"""Check the interface declared by a pretrained policy before rollout."""


def check_policy_features(policy, available):
    failures = []
    padded_policy = policy.type in {"pi0", "pi05"}
    for key, expected in {**(policy.input_features or {}), **(policy.output_features or {})}.items():
        actual = available.get(key)
        if actual is None:
            if padded_policy and key.startswith("observation.images.empty_camera_"):
                continue
            failures.append(f"{key}: missing")
            continue
        shape_ok = tuple(actual.shape) == tuple(expected.shape)
        if padded_policy and key == "observation.state":
            shape_ok = len(actual.shape) == 1 and 0 < actual.shape[0] <= policy.max_state_dim
        if padded_policy and key.startswith("observation.images."):
            shape_ok = len(actual.shape) == 3 and actual.shape[0] == expected.shape[0]
        if actual.type != expected.type or not shape_ok:
            failures.append(
                f"{key}: expected {expected.type} {expected.shape}, got {actual.type} {actual.shape}"
            )
    if failures:
        raise ValueError("MuJoCo checkpoint features differ:\n" + "\n".join(failures))
