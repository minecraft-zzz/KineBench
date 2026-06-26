import json

import numpy as np

from kinebench.io import write_json


def test_write_json_numpy(tmp_path):
    path = tmp_path / "info.json"
    write_json(path, {"arr": np.array([1, 2]), "scalar": np.float32(0.5)})
    data = json.loads(path.read_text())
    assert data == {"arr": [1, 2], "scalar": 0.5}

