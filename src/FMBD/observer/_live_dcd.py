"""Complete-frame DCD journal and atomically published viewer snapshots."""
from pathlib import Path
import errno
import os
import shutil
import struct
import tempfile

import numpy as np


def is_sharing_error(error):
    return isinstance(error, PermissionError) or getattr(error, 'winerror', None) in (5,32,33) or error.errno in (errno.EACCES,errno.EBUSY)


class LiveDCDFile:
    """Keep integration writes independent of a reader locking the public path.

    The journal is flushed and closed after every complete Cartesian frame.
    Publishing streams its bytes into a temporary file in the same directory
    and atomically replaces the public DCD. A locked public file retains the
    last readable snapshot until publication succeeds; no frames are lost from
    the journal. Memory use is independent of trajectory length.
    """
    def __init__(self, path, n_atoms, frame_dt_ps):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.journal=self.path.with_name('.'+self.path.name+'.stream')
        self.n_atoms=int(n_atoms)
        self.frame_count=0
        if self.n_atoms<=0:
            raise ValueError('DCD atom count must be positive')
        controls=[0]*20
        controls[2]=1
        controls[19]=24  # CHARMM header, no periodic unit-cell records.
        header=struct.pack('<4s9if10i',b'CORD',*controls[:9],float(frame_dt_ps)/0.04888821,*controls[10:])
        title=b'FMBD complete-frame Brownian trajectory'.ljust(80,b' ')
        with self.journal.open('wb') as handle:
            self._record(handle,header)
            self._record(handle,struct.pack('<i',1)+title)
            self._record(handle,struct.pack('<i',self.n_atoms))

    @staticmethod
    def _record(handle, data):
        marker=struct.pack('<i',len(data))
        handle.write(marker)
        handle.write(data)
        handle.write(marker)

    def append(self, coordinates):
        xyz=np.asarray(coordinates,dtype='<f4')
        if xyz.shape!=(self.n_atoms,3) or not np.isfinite(xyz).all():
            raise ValueError('DCD coordinates must be finite with shape (n_atoms,3)')
        with self.journal.open('r+b') as handle:
            handle.seek(0,2)
            boundary=handle.tell()
            try:
                for axis in range(3):
                    self._record(handle,np.ascontiguousarray(xyz[:,axis]).tobytes())
                handle.flush()
                handle.seek(8)
                handle.write(struct.pack('<i',self.frame_count+1))
                handle.seek(20)
                handle.write(struct.pack('<i',self.frame_count+1))
                handle.flush()
            except OSError:
                handle.truncate(boundary)
                handle.seek(8)
                handle.write(struct.pack('<i',self.frame_count))
                handle.seek(20)
                handle.write(struct.pack('<i',self.frame_count))
                handle.flush()
                raise
        self.frame_count+=1

    def publish(self, destination=None):
        destination=self.path if destination is None else Path(destination)
        fd,name=tempfile.mkstemp(prefix='.'+self.path.name+'.',suffix='.tmp',dir=self.path.parent)
        try:
            with os.fdopen(fd,'wb') as target,self.journal.open('rb') as source:
                shutil.copyfileobj(source,target,length=1024*1024)
                target.flush()
            try:
                os.replace(name,destination)
            except OSError as error:
                if is_sharing_error(error):
                    return False
                raise
            return True
        finally:
            if os.path.exists(name):
                os.unlink(name)
