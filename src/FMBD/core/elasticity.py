"""Linear reduced elasticity."""

from __future__ import annotations

from FMBD.core.body import Body


def linear_reduced_force(body: Body):
    return -(body.model.K @ body.state.q)


def linear_reduced_energy(body: Body):
    Kq = body.model.K @ body.state.q
    return 0.5 * body.model.xp.dot(body.state.q, Kq)


def linear_reduced_force_and_energy(body: Body, *, diagnostics: bool):
    Kq = body.model.K @ body.state.q
    energy = 0.5 * body.model.xp.dot(body.state.q, Kq) if diagnostics else None
    return -Kq, energy
