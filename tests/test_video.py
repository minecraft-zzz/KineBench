import numpy as np

from kinebench.generation.local import LocalVideoGenerator
from kinebench.video import ensure_bcthw


def test_ensure_bcthw_from_thwc():
    video = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    out = ensure_bcthw(video, target_frames=4, size=(4, 4))
    assert out.shape == (1, 3, 4, 4, 4)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_local_video_generator_npy(tmp_path):
    path = tmp_path / "video.npy"
    np.save(path, np.zeros((3, 8, 8, 3), dtype=np.uint8))
    gen = LocalVideoGenerator(path, target_frames=5, size=(4, 4))
    out = gen.generate("prompt", np.zeros((1, 3, 4, 4), dtype=np.float32))
    assert out.frames.shape == (1, 3, 5, 4, 4)
    assert out.source_path == path



def test_local_video_generator_takes_head_frames(tmp_path):
    path = tmp_path / "video.npy"
    video = np.arange(6, dtype=np.uint8).reshape(6, 1, 1, 1).repeat(3, axis=-1)
    np.save(path, video)
    gen = LocalVideoGenerator(path, target_frames=4, size=(1, 1))
    out = gen.generate("prompt", np.zeros((1, 3, 1, 1), dtype=np.float32))
    values = out.frames[0, 0, :, 0, 0]
    np.testing.assert_allclose(values, np.array([0, 1, 2, 3], dtype=np.float32) / 255.0)
