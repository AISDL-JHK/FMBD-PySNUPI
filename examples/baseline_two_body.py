"""Minimal reusable FMBD-SNUPI application template.

Before running, replace the two ``.bodyrom`` paths and the selected node pairs
with data for your experiment.  Create a ``.bodyrom`` artifact once from an
offline MOR output; the simulation runtime should load the artifact, not PySNUPI.
"""

from __future__ import annotations

import numpy as np

from FMBD import (
    Body,
    BodyModel,
    BodyState,
    DCDTrajectoryWriter,
    DynamicsOptions,
    FMBDSystem,
    IntegratorOptions,
    InterBodyBond,
    OverdampedFMBDIntegrator,
    PDBTopologyWriter,
    Simulation,
)
from FMBD.backend import to_numpy
from FMBD.interaction import DebyeHuckelInteraction, MorsePairInteraction
from FMBD.protocol import PiecewiseProtocol


def main() -> None:
    # NumPy/SciPy runs on CPU. Change this to "cuda" to use CuPy on device 0.
    backend = "cpu"
    device = 0
    dtype = np.float64

    # Offline artifacts.  Export these once with BodyModel.save(...).
    model_a = BodyModel.load(
        "body_a.bodyrom", backend=backend, device=device, dtype=dtype,
    )
    model_b = BodyModel.load(
        "body_b.bodyrom", backend=backend, device=device, dtype=dtype,
    )

    # Runtime state: q is flexible deformation; R/c are rigid pose.
    state_a = BodyState.at_reference(model_a)
    state_b = BodyState.at_reference(
        model_b,
        c=model_b.xp.asarray([10.0, 0.0, 0.0], dtype=dtype),
    )

    system = FMBDSystem()
    system.add_body("a", Body(model_a, state_a, dynamics=DynamicsOptions(pose_brownian=False, modal_brownian=True)))
    system.add_body("b", Body(model_b, state_b, dynamics=DynamicsOptions(pose_brownian=False, modal_brownian=True)))

    # Optional selected-pair attraction.  Replace with your own node pairs.
    stacking_pairs = np.array([[0, 0]], dtype=np.int64)
    system.add_interaction(MorsePairInteraction("a", "b", stacking_pairs, epsilon=42.2, a=2.475, r0=0.3831))

    # Optional generic screened electrostatic repulsion.  Supply exclusions
    # when selected nodes should not also participate in electrostatics.
    system.add_interaction(DebyeHuckelInteraction(
        "a", "b", charge=0.7,
        excluded_nodes_a=stacking_pairs[:, 0],
        excluded_nodes_b=stacking_pairs[:, 1],
    ))

    protocol = PiecewiseProtocol(
        key="Mg_mM",
        schedule=[(25.0, 10_000), (5.0, 10_000), (25.0, 10_000)],
        dt=5.0,
    )
    integrator = OverdampedFMBDIntegrator(
        dt=5.0,
        options=IntegratorOptions(max_translation_step=0.25, max_rotation_step=0.02, max_modal_step=0.50),
    )

    observers = [
        PDBTopologyWriter("output/system.pdb", inter_body_bonds=[InterBodyBond("a", 0, "b", 0)]),
        DCDTrajectoryWriter("output/trajectory.dcd", every=1_000),
    ]
    simulation = Simulation(
        system, protocol, integrator, observers=observers, output_dir="output",
    )

    def report(result) -> None:
        if (result.context.step + 1) % 1_000 == 0:
            total = sum(float(to_numpy(value)) for value in result.energies.values())
            print(f"step={result.context.step + 1:7d}  Mg={result.context.environment['Mg_mM']:4.1f} mM  E={total:.5e}")

    simulation.run(diagnostics_every=1_000, callback=report)


if __name__ == "__main__":
    main()
