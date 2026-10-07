from __future__ import annotations

import numpy as np

from FMBD.legacy_numpy_cupy import Body, BodyModel, BodyState, FMBDSystem
from FMBD.legacy_numpy_cupy.interaction import MorsePairInteraction


def one_node_model(name: str) -> BodyModel:
    dtype = np.float64
    return BodyModel(
        name=name, X=np.zeros((1, 3), dtype=dtype), Q0=np.eye(3, dtype=dtype)[None],
        phi=np.zeros((6, 0), dtype=dtype), phi_pos=np.zeros((1, 3, 0), dtype=dtype), phi_rot=np.zeros((1, 3, 0), dtype=dtype),
        K=np.zeros((0, 0), dtype=dtype), Gamma=np.zeros((0, 0), dtype=dtype), mu=np.zeros((0, 0), dtype=dtype),
        S=np.zeros((0, 0), dtype=dtype), Bq=np.zeros((0, 0), dtype=dtype),
        Z_rigid=np.eye(6, dtype=dtype), mu_rigid=np.eye(6, dtype=dtype), S_rigid=np.eye(6, dtype=dtype),
        rigid_reference_center=np.zeros(3, dtype=dtype), connectivity=np.empty((0, 2), dtype=np.int64),
    )


def test_morse_force_is_equal_and_opposite_and_zero_at_r0() -> None:
    a, b = one_node_model("a"), one_node_model("b")
    system = FMBDSystem()
    system.add_body("a", Body(a, BodyState.at_reference(a)))
    system.add_body("b", Body(b, BodyState(q=np.empty(0), R=np.eye(3), c=np.array([0.5, 0.0, 0.0]))))
    system.reconstruct_all()
    interaction = MorsePairInteraction("a", "b", np.array([[0, 0]]), epsilon=4.0, a=2.0, r0=0.5)
    result = interaction.compute(system, context=None, diagnostics=True)
    wa, wb = result.nodal_wrenches["a"], result.nodal_wrenches["b"]
    np.testing.assert_allclose(wa.force + wb.force, np.zeros((1, 3)))
    np.testing.assert_allclose(wa.force, np.zeros((1, 3)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.energy, -4.0)
