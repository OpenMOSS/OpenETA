from __future__ import annotations

import numpy as np

from tools.embodied_gateway import _anygrasp_rotation_to_panda_grip_site


def test_anygrasp_axes_are_relabelled_for_panda_grip_site() -> None:
    anygrasp_rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    panda_rotation = _anygrasp_rotation_to_panda_grip_site(anygrasp_rotation)

    assert panda_rotation is not None
    np.testing.assert_allclose(
        panda_rotation,
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    panda = np.asarray(panda_rotation)
    anygrasp = np.asarray(anygrasp_rotation)
    np.testing.assert_allclose(panda[:, 0], anygrasp[:, 1])
    np.testing.assert_allclose(panda[:, 1], anygrasp[:, 2])
    np.testing.assert_allclose(panda[:, 2], anygrasp[:, 0])
    np.testing.assert_allclose(panda.T @ panda, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(panda), 1.0, atol=1e-7)


def test_anygrasp_axis_adapter_rejects_malformed_rotation() -> None:
    assert _anygrasp_rotation_to_panda_grip_site(None) is None
    assert _anygrasp_rotation_to_panda_grip_site([[1.0]]) is None
