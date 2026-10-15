# Continuing a Brownian simulation

When `Simulation` has an `output_dir`, finalization automatically writes
`restart.pt` there. It contains body positions, rotations, NMA and residual OU
coordinates, Brownian random generator state, integration options, completed
step count and built-in beam/linker reference frames. Models and interaction
parameters remain supplied by the simulation setup; use the same inputs and
parameters when continuing. Custom interactions with persistent internal state
need to extend the checkpoint reference fields.

Configure a new simulation with a new output directory and a protocol long
enough to include both the old and additional steps, then call:

```python
simulation.load_restart(previous_output / "restart.pt")
simulation.run(n_steps=additional_steps)
```

Steps and protocol time continue from the saved endpoint. The new PDB and DCD
begin at the restored configuration; old output files are not appended to.
At the end the new directory receives another `restart.pt`.
Only load checkpoint files you trust.
