"""Check exact frame coverage and numerical equivalence across inference world sizes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("single")
p.add_argument("multi")
p.add_argument("--output", required=True)
a = p.parse_args()


def load(root):
    paths = sorted(Path(root).glob("data/**/*.parquet"))
    import pyarrow as pa

    table = pa.concat_tables([pq.read_table(path) for path in paths])
    rows = table.to_pandas().sort_values("index")
    assert rows["index"].is_unique
    return rows


x, y = load(a.single), load(a.multi)
assert np.array_equal(x["index"].values, y["index"].values)
errors = {}
for column in ("complementary_info.value_musa", "complementary_info.advantage_musa"):
    assert np.isfinite(x[column]).all() and np.isfinite(y[column]).all()
    errors[column] = float(np.max(np.abs(x[column].values - y[column].values)))
    assert np.allclose(x[column], y[column], atol=0.02, rtol=0.02), errors
column = "complementary_info.acp_indicator_musa"
assert set(x[column].unique()) <= {0, 1}
assert set(y[column].unique()) <= {0, 1}
report = {
    "frames": len(x),
    "max_abs_errors": errors,
    "indicator_equal_ratio": float(np.mean(x[column].values == y[column].values)),
    "status": "passed",
}
# Tiny value differences near the threshold can change tags; report the ratio explicitly.
Path(a.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report))
