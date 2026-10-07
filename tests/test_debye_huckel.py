from __future__ import annotations

import numpy as np

from FMBD.legacy_numpy_cupy import Body, BodyModel, BodyState, FMBDSystem
from FMBD.legacy_numpy_cupy.interaction import DebyeHuckelInteraction
from FMBD.legacy_numpy_cupy.protocol import SimulationContext


def _one_node_model(name: str) -> BodyModel:
    dtype = np.float64
    return BodyModel(
        name=name, X=np.zeros((1, 3), dtype=dtype), Q0=np.eye(3, dtype=dtype)[None],
        phi=np.zeros((6, 0), dtype=dtype), phi_pos=np.zeros((1, 3, 0), dtype=dtype), phi_rot=np.zeros((1, 3, 0), dtype=dtype),
        K=np.zeros((0, 0), dtype=dtype), Gamma=np.zeros((0, 0), dtype=dtype), mu=np.zeros((0, 0), dtype=dtype),
        S=np.zeros((0, 0), dtype=dtype), Bq=np.zeros((0, 0), dtype=dtype),
        Z_rigid=np.eye(6, dtype=dtype), mu_rigid=np.eye(6, dtype=dtype), S_rigid=np.eye(6, dtype=dtype),
        rigid_reference_center=np.zeros(3, dtype=dtype), connectivity=np.empty((0, 2), dtype=np.int64),
    )


def test_debye_huckel_force_is_equal_and_opposite() -> None:
    a, b = _one_node_model("a"), _one_node_model("b")
    system = FMBDSystem()
    system.add_body("a", Body(a, BodyState.at_reference(a)))
    system.add_body("b", Body(b, BodyState(np.empty(0), np.eye(3), np.array([0.5, 0.0, 0.0]))))
    system.reconstruct_all()
    interaction = DebyeHuckelInteraction("a", "b", cutoff_multiplier=4.0)
    context = SimulationContext(step=0, time=0.0, dt=1.0, environment={"Mg_mM": 25.0})
    interaction.initialize(system, context)
    result = interaction.compute(system, context, diagnostics=True)
    wa, wb = result.nodal_wrenches["a"], result.nodal_wrenches["b"]
    np.testing.assert_allclose(wa.force + wb.force, np.zeros((1, 3)))
    assert result.energy is not None and result.energy.item() > 0.0
    assert result.diagnostics["n_used"] == 1


def test_gpu_debye_huckel_matches_cpu_neighbor_list_and_force() -> None:
    import pytest

    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def evaluate(backend):
        a, b = _one_node_model("a"), _one_node_model("b")
        if backend == "cuda":
            a = BodyModel.from_dict(a.to_dict(), backend="cuda")
            b = BodyModel.from_dict(b.to_dict(), backend="cuda")
            xp = cp
        else:
            xp = np
        system = FMBDSystem()
        system.add_body("a", Body(a, BodyState.at_reference(a)))
        system.add_body("b", Body(b, BodyState(xp.empty(0), xp.eye(3), xp.asarray([0.5, 0.0, 0.0]))))
        system.reconstruct_all()
        interaction = DebyeHuckelInteraction("a", "b", cutoff_multiplier=4.0)
        context = SimulationContext(step=0, time=0.0, dt=1.0, environment={"Mg_mM": 25.0})
        interaction.initialize(system, context)
        result = interaction.compute(system, context, diagnostics=True)
        return interaction._pairs, result.nodal_wrenches["a"].force

    cpu_pairs, cpu_force = evaluate("cpu")
    gpu_pairs, gpu_force = evaluate("cuda")
    np.testing.assert_array_equal(cp.asnumpy(gpu_pairs), cpu_pairs)
    np.testing.assert_allclose(cp.asnumpy(gpu_force), cpu_force, rtol=1e-12, atol=1e-12)
