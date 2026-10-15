"""Portable runtime checkpoints for a simulation configured with the same models."""
from dataclasses import fields, is_dataclass, replace, asdict
from pathlib import Path
import os
import torch


def _convert(value, device=None, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device or 'cpu', dtype=dtype if value.is_floating_point() else value.dtype).clone()
    if isinstance(value, dict):
        return {key: _convert(item, device, dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_convert(item, device, dtype) for item in value)
    if is_dataclass(value):
        return replace(value, **{field.name: _convert(getattr(value, field.name), device, dtype) for field in fields(value)})
    return value


def save_restart(simulation, path):
    """Save state, RNG, solver options and persistent interaction references.

    Restore into a Simulation with the same body models and interactions.
    Protocols may be extended to cover the requested subsequent steps.
    """
    reference_fields = ('_frames', '_frame_a', '_frame_b', '_references',
                        '_rotation_a0', '_rotation_b0', '_axis')
    data = dict(schema='fmbd.restart', version=1,
        completed_steps=simulation.last_completed_step or 0,
        dt=simulation.integrator.dt, options=asdict(simulation.integrator.options),
        bodies={name: dict(model_name=body.model.name, n_node=body.model.n_node,
            n_mode=body.model.n_mode, n_respod_mode=body.model.n_respod_mode,
            dynamics=asdict(body.dynamics), dynamic=body.dynamic,
            state=_convert({key:getattr(body.state,key) for key in ('c','R','q','a')}))
            for name,body in simulation.system.bodies.items()},
        interactions=[dict(type=f'{type(item).__module__}.{type(item).__qualname__}',
            name=item.name, body_a=getattr(item,'body_a',None),body_b=getattr(item,'body_b',None),
            references=_convert({key:getattr(item,key) for key in reference_fields if hasattr(item,key)}))
            for item in simulation.system.interactions],
        generator_state=None if simulation.integrator.generator is None else simulation.integrator.generator.get_state().cpu(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(data,temporary)
    os.replace(temporary,path)


def load_restart(simulation, path):
    """Restore a trusted checkpoint before running an extended protocol."""
    if simulation._initialized:
        raise RuntimeError('Load restart before initializing the simulation')
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('schema')!='fmbd.restart' or data.get('version')!=1:
        raise ValueError('Unsupported FMBD restart file')
    if data['dt']!=simulation.integrator.dt or data['options']!=asdict(simulation.integrator.options):
        raise ValueError('Restart integrator settings differ')
    if list(data['bodies'])!=list(simulation.system.bodies):
        raise ValueError('Restart body names/order differ')
    if len(data['interactions'])!=len(simulation.system.interactions):
        raise ValueError('Restart interaction count differs')
    for name,body in simulation.system.bodies.items():
        saved=data['bodies'][name]
        if (saved['model_name'],saved['n_node'],saved['n_mode'],saved['n_respod_mode'])!=(body.model.name,body.model.n_node,body.model.n_mode,body.model.n_respod_mode):
            raise ValueError(f'Restart body model differs: {name}')
    for item,saved in zip(simulation.system.interactions,data['interactions']):
        signature=(f'{type(item).__module__}.{type(item).__qualname__}',item.name,getattr(item,'body_a',None),getattr(item,'body_b',None))
        if signature!=(saved['type'],saved['name'],saved['body_a'],saved['body_b']):
            raise ValueError('Restart interaction identity differs')
    for name,body in simulation.system.bodies.items():
        saved=data['bodies'][name]
        for key,value in _convert(saved['state'],body.model.X.device,body.model.X.dtype).items():
            setattr(body.state,key,value)
        body.dynamic=saved['dynamic']
        body.dynamics=type(body.dynamics)(**saved['dynamics'])
    simulation.last_completed_step=data['completed_steps']
    simulation.initialize()
    first=next(iter(simulation.system.bodies.values())).model.X
    for item,saved in zip(simulation.system.interactions,data['interactions']):
        tensor=simulation.system.bodies[getattr(item,'body_a',next(iter(simulation.system.bodies)))].model.X
        for key,value in _convert(saved['references'],tensor.device,tensor.dtype).items():
            setattr(item,key,value)
    if data['generator_state'] is not None:
        if simulation.integrator.generator is None:
            simulation.integrator.generator=torch.Generator(device=first.device)
        simulation.integrator.generator.set_state(data['generator_state'])
    torch.set_rng_state(data['cpu_rng'])
    if data['cuda_rng'] and first.is_cuda:
        for index,state in enumerate(data['cuda_rng'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state,index)
    return data
