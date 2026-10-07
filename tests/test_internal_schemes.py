from __future__ import annotations

import numpy as np
import pickle

from FMBD.legacy_numpy_cupy import (
    Body,
    BodyModel,
    BodyState,
    DynamicsOptions,
    FMBDSystem,
    GeneralizedForce,
    IntegratorOptions,
    OverdampedFMBDIntegrator,
    reconstruct,
)


def _closure_model() -> BodyModel:
    dtype = np.float64
    phi = np.zeros((12, 1), dtype=dtype)
    phi[0, 0] = 1.0
    psi = np.zeros((12, 1), dtype=dtype)
    psi[1, 0] = 1.0
    return BodyModel(
        name="closure-test",
        X=np.zeros((2, 3), dtype=dtype),
        Q0=np.broadcast_to(np.eye(3, dtype=dtype), (2, 3, 3)).copy(),
        phi=phi,
        phi_pos=phi.reshape(2, 6, 1)[:, :3],
        phi_rot=phi.reshape(2, 6, 1)[:, 3:],
        K=np.eye(1, dtype=dtype), Gamma=np.eye(1, dtype=dtype),
        mu=np.eye(1, dtype=dtype), S=np.eye(1, dtype=dtype),
        Bq=np.eye(1, dtype=dtype),
        Z_rigid=np.eye(6, dtype=dtype), mu_rigid=np.eye(6, dtype=dtype),
        S_rigid=np.eye(6, dtype=dtype),
        rigid_reference_center=np.zeros(3, dtype=dtype),
        connectivity=np.array([[0, 1]], dtype=np.int64),
        modal_mean_force=np.ones(1, dtype=dtype),
        respod_psi=psi,
        respod_ou_rho=np.array([0.5], dtype=dtype),
        respod_ou_sigma=np.array([2.0], dtype=dtype),
        closure_dt_ps=1.0,
    )


def test_meanforce_respod_ou_applies_force_and_exact_discrete_ou() -> None:
    model = _closure_model()
    state = BodyState.at_reference(model)
    assert state.a is not None
    state.a.fill(1.0)
    body = Body(model, state, dynamics=DynamicsOptions(pose_brownian=False, modal_brownian=False))
    system = FMBDSystem()
    system.add_body("body", body)
    force = GeneralizedForce.zeros(1, xp=np, dtype=np.float64)
    integrator = OverdampedFMBDIntegrator(1.0, IntegratorOptions(internal_scheme="meanforce_respod_ou"))
    integrator.step(system, {"body": force}, context=None)
    np.testing.assert_allclose(state.q, [1.0])
    np.testing.assert_allclose(state.a, [0.5])
    reconstructed = reconstruct(body)
    np.testing.assert_allclose(reconstructed.U[0, :2], [1.0, 0.5])


def test_respod_scheme_rejects_mismatched_fitting_interval() -> None:
    model = _closure_model()
    body = Body(model, BodyState.at_reference(model), dynamics=DynamicsOptions(pose_brownian=False, modal_brownian=False))
    system = FMBDSystem()
    system.add_body("body", body)
    force = GeneralizedForce.zeros(1, xp=np, dtype=np.float64)
    integrator = OverdampedFMBDIntegrator(2.0, IntegratorOptions(internal_scheme="meanforce_respod_ou"))
    try:
        integrator.step(system, {"body": force}, context=None)
    except ValueError as err:
        assert "closure_ou_dt_ps" in str(err)
    else:
        raise AssertionError("mismatched OU fitting interval must raise ValueError")


def test_load_snupy_fmbd_data_schema(tmp_path) -> None:
    model = _closure_model()
    source = {
        "schema": "snupy.mor_fmbd_input", "name": "prepared", "closure_enabled": True,
        "init_node": np.concatenate((model.X, np.zeros((2, 3))), axis=1),
        "phi": model.phi, "K_r": model.K, "Gamma_r": model.Gamma,
        "mu_r": model.mu, "S_r": model.S, "Bq": model.Bq,
        "Z_rigid": model.Z_rigid, "mu_rigid": model.mu_rigid,
        "S_rigid": model.S_rigid, "rigid_reference_center": model.rigid_reference_center,
        "e_conn": model.connectivity, "r_pre_r": model.modal_mean_force,
        "respod_psi": model.respod_psi, "respod_ou_rho": model.respod_ou_rho,
        "respod_ou_sigma": model.respod_ou_sigma, "closure_ou_dt_ps": 1.0,
    }
    path = tmp_path / "FMBD_data.pkl"
    with path.open("wb") as fh:
        pickle.dump(source, fh)
    loaded = BodyModel.from_fmbd_data(path)
    assert loaded.n_mode == 1
    assert loaded.n_respod_mode == 1
