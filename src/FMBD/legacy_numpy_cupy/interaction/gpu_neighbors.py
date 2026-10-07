"""CuPy cell-list search for cross-body Verlet pairs."""

from __future__ import annotations

import numpy as np


_SOURCE = r'''
extern "C" {
__device__ long long find_cell(const long long* keys, const long long size, const long long key) {
    long long left = 0, right = size;
    while (left < right) {
        const long long middle = left + (right-left)/2;
        if (keys[middle] < key) left = middle+1; else right = middle;
    }
    return (left < size && keys[left] == key) ? left : -1;
}
__device__ long long linear_cell(const long long x, const long long y, const long long z,
                                 const long long nx, const long long ny) {
    return x + nx*(y + ny*z);
}
__global__ void count_pairs(
    const REAL* a, const long long* cell_a, const unsigned char* active_a,
    const REAL* b, const long long* sorted_b, const long long* cell_keys,
    const long long* cell_starts, const long long* cell_counts,
    const unsigned char* active_b, const long long na, const long long key_count,
    const long long nx, const long long ny, const long long nz,
    const REAL cutoff2, long long* counts) {
    const long long i = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= na) return;
    if (!active_a[i]) { counts[i] = 0; return; }
    const long long cx=cell_a[3*i], cy=cell_a[3*i+1], cz=cell_a[3*i+2];
    long long total=0;
    for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
        const long long x=cx+dx, y=cy+dy, z=cz+dz;
        if (x<0 || y<0 || z<0 || x>=nx || y>=ny || z>=nz) continue;
        const long long location=find_cell(cell_keys,key_count,linear_cell(x,y,z,nx,ny));
        if (location<0) continue;
        const long long begin=cell_starts[location], end=begin+cell_counts[location];
        for (long long k=begin; k<end; ++k) {
            const long long j=sorted_b[k]; if (!active_b[j]) continue;
            const REAL dxp=b[3*j]-a[3*i], dyp=b[3*j+1]-a[3*i+1], dzp=b[3*j+2]-a[3*i+2];
            total += (dxp*dxp+dyp*dyp+dzp*dzp <= cutoff2);
        }
    }
    counts[i]=total;
}
__global__ void fill_pairs(
    const REAL* a, const long long* cell_a, const unsigned char* active_a,
    const REAL* b, const long long* sorted_b, const long long* cell_keys,
    const long long* cell_starts, const long long* cell_counts,
    const unsigned char* active_b, const long long* offsets, const long long na,
    const long long key_count, const long long nx, const long long ny, const long long nz,
    const REAL cutoff2, long long* pairs) {
    const long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=na || !active_a[i]) return;
    const long long cx=cell_a[3*i], cy=cell_a[3*i+1], cz=cell_a[3*i+2];
    long long out=offsets[i];
    for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
        const long long x=cx+dx, y=cy+dy, z=cz+dz;
        if (x<0 || y<0 || z<0 || x>=nx || y>=ny || z>=nz) continue;
        const long long location=find_cell(cell_keys,key_count,linear_cell(x,y,z,nx,ny));
        if (location<0) continue;
        const long long begin=cell_starts[location], end=begin+cell_counts[location];
        for (long long k=begin; k<end; ++k) {
            const long long j=sorted_b[k]; if (!active_b[j]) continue;
            const REAL dxp=b[3*j]-a[3*i], dyp=b[3*j+1]-a[3*i+1], dzp=b[3*j+2]-a[3*i+2];
            if (dxp*dxp+dyp*dyp+dzp*dzp <= cutoff2) {
                pairs[2*out]=i; pairs[2*out+1]=j; ++out;
            }
        }
    }
}
}
'''


class GPUCrossBodyPairSearch:
    """Build an exact cross-body radius list with a GPU uniform cell list."""

    def __init__(self, *, device: int = 0):
        from FMBD.legacy_numpy_cupy.backend import cupy_module

        self.cp = cupy_module()
        self.device = int(device)
        self._modules = {}

    def _kernels(self, dtype):
        dtype = np.dtype(dtype)
        key = dtype.str
        if key not in self._modules:
            real = "float" if dtype == np.dtype(np.float32) else "double"
            module = self.cp.RawModule(
                code=_SOURCE.replace("REAL", real), options=("--std=c++11",),
            )
            self._modules[key] = (module.get_function("count_pairs"), module.get_function("fill_pairs"))
        return self._modules[key]

    def query(self, xa, xb, cutoff: float, excluded_a=(), excluded_b=()):
        with self.cp.cuda.Device(self.device):
            return self._query(xa, xb, cutoff, excluded_a, excluded_b)

    def _query(self, xa, xb, cutoff: float, excluded_a=(), excluded_b=()):
        cp = self.cp
        if xa.ndim != 2 or xb.ndim != 2 or xa.shape[1:] != (3,) or xb.shape[1:] != (3,):
            raise ValueError("cross-body coordinates must have shape (N, 3)")
        if xa.dtype != xb.dtype or xa.dtype not in (cp.float32, cp.float64):
            raise TypeError("GPU cell-list coordinates must share float32 or float64 dtype")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        xa, xb = cp.ascontiguousarray(xa), cp.ascontiguousarray(xb)
        na, nb = len(xa), len(xb)
        if na == 0 or nb == 0:
            return cp.empty((0, 2), dtype=cp.int64)
        # Share one origin so both clouds use the same cell coordinates.
        origin = cp.minimum(cp.min(xa, axis=0), cp.min(xb, axis=0))
        width = float(cutoff)
        cells_a = cp.floor((xa-origin)/width).astype(cp.int64)
        cells_b = cp.floor((xb-origin)/width).astype(cp.int64)
        dimensions = cp.maximum(cp.max(cp.concatenate((cells_a, cells_b), axis=0), axis=0)+1, 1)
        nx, ny, nz = (int(v.item()) for v in dimensions)
        keys_b = cells_b[:, 0] + nx*(cells_b[:, 1] + ny*cells_b[:, 2])
        order = cp.argsort(keys_b)
        sorted_keys = keys_b[order]
        cell_keys, starts, counts_in_cell = cp.unique(sorted_keys, return_index=True, return_counts=True)
        active_a = cp.ones(na, dtype=cp.uint8)
        active_b = cp.ones(nb, dtype=cp.uint8)
        if len(excluded_a):
            active_a[cp.asarray(excluded_a, dtype=cp.int64)] = 0
        if len(excluded_b):
            active_b[cp.asarray(excluded_b, dtype=cp.int64)] = 0
        counts = cp.empty(na, dtype=cp.int64)
        count_kernel, fill_kernel = self._kernels(xa.dtype)
        block = 128
        grid = ((na+block-1)//block,)
        shared = (xa, cells_a, active_a, xb, order, cell_keys, starts, counts_in_cell,
                  active_b, np.int64(na), np.int64(len(cell_keys)), np.int64(nx), np.int64(ny),
                  np.int64(nz), xa.dtype.type(width*width))
        count_kernel(grid, (block,), (*shared, counts))
        offsets = cp.cumsum(counts, dtype=cp.int64)-counts
        total = int((offsets[-1]+counts[-1]).item())
        pairs = cp.empty((total, 2), dtype=cp.int64)
        if total:
            fill_kernel(grid, (block,), (*shared[:9], offsets, *shared[9:], pairs))
        return pairs


__all__ = ["GPUCrossBodyPairSearch"]
