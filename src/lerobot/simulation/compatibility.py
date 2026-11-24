"""Check the interface declared by a pretrained policy before rollout."""


def check_policy_features(policy, available):
    failures = []
    for key, expected in {**policy.input_features, **policy.output_features}.items():
        actual = available.get(key)
        if actual is None:
            failures.append(f"{key}: missing")
        elif actual.type != expected.type or tuple(actual.shape) != tuple(expected.shape):
            failures.append(
                f"{key}: expected {expected.type} {expected.shape}, got {actual.type} {actual.shape}"
            )
    if failures:
        raise ValueError("MuJoCo checkpoint features differ:\n" + "\n".join(failures))
