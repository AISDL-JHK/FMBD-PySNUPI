# Freely rotating ssDNA tether

`SingleStrandTetherInteraction` applies central forces at the supplied node
pairs. It does not constrain node orientations or impose endpoint couples.
Attachment offsets still generate body torque through force projection.

For extension fraction `x = r/Lc`, use the inverse-Langevin rational
approximation `f = (kBT/b) x(3-x²)/(1-x²)`, where `b` is the effective Kuhn
length. This is an FJC approximation, not an exact inverse or an empirical
fit to short ssDNA. The corresponding vector-coordinate potential is
`U = (kBT Lc/b) [x²/2 - log(1-x²)]`, with `U(0)=0`.
The radial sampling distribution also includes the geometric `r²` measure.

Above `regularization_fraction=0.98`, force is continued linearly using the
tangent at the threshold; potential uses the matching quadratic continuation.
This numerical safeguard does not model physical overstretching. Diagnostics
include extension fractions and a regularization flag; applications should
monitor these and reject contour-length violations.

`FJCTetherParameters.from_nucleotides(16, kuhn_length=1.5)` retains the existing
attachment-inclusive contour convention (11.3061 nm for 16 unpaired nt).
The default Kuhn length 1.5 nm and temperature 300 K are explicit modeling
assumptions, not calibrated values. Beam bending and torsion parameters are
not reused. Compatible tethers across bodies are evaluated together without
autograd, quadrature or iterative inverse-Langevin solves.
