from __future__ import annotations

import pickle

import numpy as np
import torch

from FMBD import (
    Body,
    BodyModel,
    BodyState,
    DynamicsOptions,
    FMBDSystem,
    FunctionalInteraction,
    GeneralizedForce,
    IntegratorOptions,
    InteractionResult,
    OverdampedFMBDIntegrator,
    PiecewiseProtocol,
    Simulation,
)


def _rigid_test_model() -> BodyModel:
    dtype = torch.float64
    return BodyModel(
        name="rigid-test",
        X=torch.zeros((1, 3), dtype=dtype),
        Q0=torch.eye(3, dtype=dtype)[None],
        phi=torch.zeros((6, 0), dtype=dtype),
        phi_pos=torch.zeros((1, 3, 0), dtype=dtype),
        phi_rot=torch.zeros((1, 3, 0), dtype=dtype),
        K=torch.zeros((0, 0), dtype=dtype),
        Gamma=torch.zeros((0, 0), dtype=dtype),
        mu=torch.zeros((0, 0), dtype=dtype),
        S=torch.zeros((0, 0), dtype=dtype),
        Bq=torch.zeros((0, 0), dtype=dtype),
        Z_rigid=torch.eye(6, dtype=dtype),
        mu_rigid=torch.eye(6, dtype=dtype),
        S_rigid=torch.eye(6, dtype=dtype),
        rigid_reference_center=torch.zeros(3, dtype=dtype),
        connectivity=np.empty((0, 2), dtype=np.int64),
    )


def _linear_translation_simulation(scheme: str):
    model = _rigid_test_model()
    state = BodyState.at_reference(model, c=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    body = Body(
        model,
        state,
        dynamics=DynamicsOptions(pose_brownian=False, modal_brownian=False),
    )
    system = FMBDSystem()
    system.add_body("body", body)
    evaluations: list[float] = []

    def linear_force(current_system, context, diagnostics):
        current = current_system.bodies["body"].state.c
        evaluations.append(float(current[0]))
        force = GeneralizedForce(
            translation=-current.clone(),
            rotation=current.new_zeros(3),
            modal=current.new_zeros(0),
        )
        return InteractionResult(generalized_forces={"body": force})

    system.add_interaction(FunctionalInteraction(linear_force))
    protocol = PiecewiseProtocol(key="unused", schedule=[(0.0, 1)], dt=0.2)
    integrator = OverdampedFMBDIntegrator(
        0.2,
        IntegratorOptions(scheme=scheme),
    )
    simulation = Simulation(system, protocol, integrator)
    simulation.step(0)
    return state, evaluations


def test_midpoint_reevaluates_force_at_half_step() -> None:
    state, evaluations = _linear_translation_simulation("midpoint_brownian")
    torch.testing.assert_close(state.c, torch.tensor([0.82, 0.0, 0.0], dtype=torch.float64))
    np.testing.assert_allclose(evaluations, [1.0, 0.9])


def test_euler_brownian_remains_available() -> None:
    state, evaluations = _linear_translation_simulation("euler_brownian")
    torch.testing.assert_close(state.c, torch.tensor([0.8, 0.0, 0.0], dtype=torch.float64))
    np.testing.assert_allclose(evaluations, [1.0])


def test_load_current_pysnupi_fmbd_schema(tmp_path) -> None:
    node_count, nma_modes, pod_modes = 2, 1, 1
    phi = np.zeros((6 * node_count, nma_modes), dtype=np.float64)
    phi[0, 0] = 1.0
    psi = np.zeros((6 * node_count, pod_modes), dtype=np.float64)
    psi[1, 0] = 1.0
    source = {
        "schema": "snupy.mor_fmbd_input",
        "version": 1,
        "name": "pysnupi-export",
        "init_node": np.zeros((node_count, 6), dtype=np.float64),
        "phi": phi,
        "K_r": np.eye(nma_modes),
        "Gamma_r": np.eye(nma_modes),
        "mu_r": np.eye(nma_modes),
        "S_r": np.eye(nma_modes),
        "Bq": np.eye(nma_modes),
        "Z_rigid": np.eye(6),
        "mu_rigid": np.eye(6),
        "S_rigid": np.eye(6),
        "rigid_reference_center": np.zeros(3),
        "e_conn": np.array([[0, 1]], dtype=np.int64),
        "r_pre_r": np.ones(nma_modes),
        "closure_enabled": True,
        "respod_psi": psi,
        "respod_ou_rho": np.array([0.5]),
        "respod_ou_sigma": np.array([2.0]),
        "closure_ou_dt_ps": 5.0,
        "source_records": [],
        "units": "nm, pN, ps, rad",
    }
    path = tmp_path / "FMBD_data.pkl"
    with path.open("wb") as stream:
        pickle.dump(source, stream, protocol=pickle.HIGHEST_PROTOCOL)

    model = BodyModel.from_fmbd_data(path, device="cpu", dtype=torch.float64)

    assert model.name == "pysnupi-export"
    assert model.n_node == node_count
    assert model.n_mode == nma_modes
    assert model.n_respod_mode == pod_modes
    assert model.closure_dt_ps == 5.0
    torch.testing.assert_close(model.modal_mean_force, torch.ones(nma_modes, dtype=torch.float64))
