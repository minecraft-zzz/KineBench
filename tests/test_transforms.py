import numpy as np
import pytest

from kinebench.planning.transforms import poses7_to_T, poses_wxyz_to_xyz_euler


def test_poses7_to_T_shape_and_translation():
    poses = np.array([[1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mats = poses7_to_T(poses)
    assert mats.shape == (1, 4, 4)
    np.testing.assert_allclose(mats[0, :3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mats[0, :3, :3], np.eye(3), atol=1e-6)


def test_poses_wxyz_to_xyz_euler_shape():
    poses = np.array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    actions = poses_wxyz_to_xyz_euler(poses)
    assert actions.shape == (1, 6)
    np.testing.assert_allclose(actions[0, :3], [0.1, 0.2, 0.3], atol=1e-6)


def test_pose_validation():
    with pytest.raises(ValueError):
        poses7_to_T(np.zeros((2, 6), dtype=np.float32))

