# Examples

`local_smoke.yaml` expects `examples/local_video.npy`, shaped as either `[T,H,W,3]` or `[1,3,T,H,W]`.

Create a tiny synthetic file for plumbing tests:

```bash
python - <<'PY'
import numpy as np
v = np.zeros((4, 128, 128, 3), dtype=np.uint8)
v[..., 0] = 180
np.save("examples/local_video.npy", v)
PY
```

Then run:

```bash
python scripts/run_eval.py --config configs/eval/local_smoke.yaml
```

