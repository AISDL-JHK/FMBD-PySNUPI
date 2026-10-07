"""Torch-based flexible multibody Brownian dynamics runtime.

The primary :mod:`FMBD` API uses Torch tensors on CPU or CUDA. The former
NumPy/CuPy implementation remains available as :mod:`FMBD.legacy_numpy_cupy`
for archived workflows and result reproduction.
"""

from .core import (
    Body, BodyModel, BodyState, DynamicsOptions, FMBDSystem, GeneralizedForce,
    NodalWrench, ReconstructedBody, project_nodal_wrench, reconstruct,
)
from .dynamics import IntegratorOptions, OverdampedFMBDIntegrator
from .interaction import (
    CollectiveAxialTorsionInteraction,
    DebyeHuckelInteraction,
    EulerBernoulliBeamInteraction,
    FunctionalInteraction,
    Interaction,
    InteractionResult,
    MorsePairInteraction,
    mgcl2_ionic_strength_mM,
)
from .observer import (
    DCDTrajectoryWriter, InterBodyBond, Observer, PDBTopologyWriter,
    SimulationLogWriter,
)
from .protocol import PiecewiseProtocol, SimulationContext
from .simulation import Simulation, StepResult

__all__ = [
    "Body",
    "BodyModel",
    "BodyState",
    "CollectiveAxialTorsionInteraction",
    "DCDTrajectoryWriter",
    "DebyeHuckelInteraction",
    "DynamicsOptions",
    "EulerBernoulliBeamInteraction",
    "FMBDSystem",
    "FunctionalInteraction",
    "GeneralizedForce",
    "IntegratorOptions",
    "InterBodyBond",
    "Interaction",
    "InteractionResult",
    "MorsePairInteraction",
    "NodalWrench",
    "Observer",
    "OverdampedFMBDIntegrator",
    "PDBTopologyWriter",
    "PiecewiseProtocol",
    "ReconstructedBody",
    "Simulation",
    "SimulationContext",
    "SimulationLogWriter",
    "StepResult",
    "mgcl2_ionic_strength_mM",
    "project_nodal_wrench",
    "reconstruct",
]
