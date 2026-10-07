from __future__ import annotations

import numpy as np

from FMBD.legacy_numpy_cupy import (
    Body, BodyModel, BodyState, FMBDSystem, NodalWrench,
    project_nodal_wrench, reconstruct,
)
from FMBD.legacy_numpy_cupy.math import exp_so3, log_so3


def make_model() -> BodyModel:
    dtype = np.float64
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    phi = np.zeros((12, 1), dtype=dtype)
    phi[0, 0] = 1.0
    return BodyModel(
        name="test", X=X, Q0=np.broadcast_to(np.eye(3, dtype=dtype), (2, 3, 3)).copy(),
        phi=phi, phi_pos=phi.reshape(2, 6, 1)[:, :3], phi_rot=phi.reshape(2, 6, 1)[:, 3:],
        K=np.eye(1, dtype=dtype), Gamma=np.eye(1, dtype=dtype), mu=np.eye(1, dtype=dtype),
        S=np.ones((1, 1), dtype=dtype), Bq=np.ones((1, 1), dtype=dtype),
        Z_rigid=np.eye(6, dtype=dtype), mu_rigid=np.eye(6, dtype=dtype), S_rigid=np.eye(6, dtype=dtype),
        rigid_reference_center=np.zeros(3, dtype=dtype), connectivity=np.array([[0, 1]], dtype=np.int64),
    )


def test_reconstruction_separates_modal_and_pose() -> None:
    model = make_model()
    R = exp_so3(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
    body = Body(model, BodyState(q=np.array([2.0]), R=R, c=np.array([3.0, 4.0, 0.0])))
    rec = reconstruct(body)
    np.testing.assert_allclose(rec.y, [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(rec.x, [[3.0, 6.0, 0.0], [3.0, 5.0, 0.0]])
    assert body.reconstruction is rec


def test_versioned_schema_round_trip(tmp_path) -> None:
    model = make_model()
    path = tmp_path / "model.bodyrom"
    model.save(path)
    loaded = BodyModel.load(path)
    assert loaded.name == model.name
    np.testing.assert_allclose(loaded.phi, model.phi)
    np.testing.assert_array_equal(loaded.connectivity, model.connectivity)


def test_so3_log_exp_round_trip() -> None:
    v = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    np.testing.assert_allclose(log_so3(exp_so3(v)), v, rtol=1e-10, atol=1e-10)


def test_system_reconstructs_any_number_of_bodies() -> None:
    model = make_model()
    system = FMBDSystem()
    system.add_body("a", Body(model, BodyState.at_reference(model)))
    system.add_body("b", Body(model, BodyState.at_reference(model)))
    system.add_body("c", Body(model, BodyState.at_reference(model)))
    system.reconstruct_all()
    assert all(body.reconstruction is not None for body in system.bodies.values())


def test_projection_returns_translation_rotation_and_modal_force() -> None:
    model = make_model()
    body = Body(model, BodyState.at_reference(model))
    reconstruct(body)
    wrench = NodalWrench(
        force=np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        moment=np.zeros((2, 3)),
    )
    generalized = project_nodal_wrench(body, wrench)
    np.testing.assert_allclose(generalized.translation, [2.0, 3.0, 0.0])
    np.testing.assert_allclose(generalized.rotation, [0.0, 0.0, 3.0])
    np.testing.assert_allclose(generalized.modal, [2.0])
