from types import SimpleNamespace
import os
import numpy as np
import pytest
import torch
from FMBD.observer._live_dcd import LiveDCDFile
from FMBD.observer.trajectory import DCDTrajectoryWriter, PDBTopologyWriter, InterBodyBond


def read_frames(path):
    mda=pytest.importorskip('MDAnalysis')
    from MDAnalysis.coordinates.DCD import DCDReader
    with DCDReader(str(path)) as reader:
        return np.stack([ts.positions.copy() for ts in reader])


def test_live_read_and_locked_publication(tmp_path,monkeypatch):
    path=tmp_path/'trajectory.dcd'
    stream=LiveDCDFile(path,2,5.)
    xyz=np.array([[1.,2.,3.],[4.,5.,6.]])
    stream.append(xyz)
    assert stream.publish()
    np.testing.assert_allclose(read_frames(path),xyz[None])
    replace=os.replace
    def deny(*args):
        raise PermissionError('viewer holds an exclusive lock')
    monkeypatch.setattr(os,'replace',deny)
    stream.append(xyz+1)
    assert not stream.publish()
    np.testing.assert_allclose(read_frames(path),xyz[None])
    assert len(read_frames(stream.journal))==2
    monkeypatch.setattr(os,'replace',replace)
    stream.append(xyz+2)
    assert stream.publish()
    np.testing.assert_allclose(read_frames(path),np.stack([xyz,xyz+1,xyz+2]))
    assert not list(tmp_path.glob('*.tmp'))


def test_observer_final_frame_units_and_linker_topology(tmp_path):
    bodies={}
    for index,name in enumerate(('a','b')):
        model=SimpleNamespace(n_node=1,connectivity=np.empty((0,2),dtype=int),metadata={})
        body=SimpleNamespace(model=model,reconstruction=SimpleNamespace(
            x=torch.tensor([[float(index),0.,0.]],dtype=torch.float64)))
        bodies[name]=body
    sim=SimpleNamespace(system=SimpleNamespace(bodies=bodies),last_completed_step=None)
    context=SimpleNamespace(dt=5.)
    pdb=tmp_path/'system.pdb'
    PDBTopologyWriter(pdb,inter_body_bonds=[InterBodyBond('a',0,'b',0)]).on_start(sim,context)
    assert 'CONECT    1    2' in pdb.read_text()
    writer=DCDTrajectoryWriter(tmp_path/'trajectory.dcd',every=2,publish_every=1)
    writer.on_start(sim,context)
    assert len(read_frames(writer.path))==1
    for step in range(5):
        for body in bodies.values():
            body.reconstruction.x[:,0]+=1
        sim.last_completed_step=step+1
        writer.on_step(sim,SimpleNamespace(context=SimpleNamespace(step=step)))
    writer.on_finish(sim)
    frames=read_frames(writer.path)
    np.testing.assert_allclose(frames[:,0,0],[0,20,40,50])
    assert len(frames)==4


@pytest.mark.skipif(os.name!='nt',reason='Windows sharing-mode regression')
def test_real_windows_reader_lock(tmp_path):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    close=kernel.CloseHandle
    close.argtypes=[wintypes.HANDLE]
    path=tmp_path/'live.dcd'
    writer=LiveDCDFile(path,1,5.)
    writer.append(np.zeros((1,3)))
    assert writer.publish()
    handle=create(str(path),0x80000000,1,None,3,0,None)
    assert handle!=ctypes.c_void_p(-1).value
    try:
        writer.append(np.ones((1,3)))
        assert not writer.publish()
        assert writer.frame_count==2
        assert len(read_frames(path))==1
        recovery=tmp_path/'live_complete.dcd'
        assert writer.publish(recovery)
        assert len(read_frames(recovery))==2
    finally:
        close(handle)
    assert writer.publish()
    assert len(read_frames(path))==2
