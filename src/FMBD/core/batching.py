"""Size/device groups with shared or cached body-model matrices."""
import torch


class BodyBatch:
    def __init__(self, entries):
        self.names=tuple(name for name,_ in entries)
        self.bodies=tuple(body for _,body in entries)
        self.shared=all(body.model is self.bodies[0].model for body in self.bodies)
        self._cache={}

    def model_tensor(self, name):
        values=[getattr(body.model,name) for body in self.bodies]
        if self.shared:
            return values[0]
        signature=tuple((id(v),v._version) for v in values)
        cached=self._cache.get(name)
        if cached is None or cached[0]!=signature:
            cached=(signature,torch.stack(values))
            self._cache[name]=cached
        return cached[1]

    def apply(self, name, vectors):
        matrix=self.model_tensor(name)
        if self.shared:
            # Zero-stride expansion shares storage; batched GEMV avoids the
            # unfavorable skinny-GEMM layout of vectors @ tall_matrix.T.
            matrix=matrix.expand(len(self.bodies),*matrix.shape)
        return torch.bmm(matrix,vectors.unsqueeze(-1)).squeeze(-1)

    def state(self, name):
        return torch.stack([getattr(body.state,name) for body in self.bodies])

    def model_vectors(self, name):
        value=self.model_tensor(name)
        return value.unsqueeze(0) if self.shared else value

    def assign(self, **values):
        for index,body in enumerate(self.bodies):
            for name,value in values.items():
                setattr(body.state,name,value[index])


def grouped_bodies(system, *, dynamic_only=False, rigid=False):
    # Public bodies/dynamic flags can change, so validate the lightweight plan
    # on every use. Cached matrices are refreshed after tensor replacement or
    # in-place modification. State tensors are always gathered fresh.
    entries=[(name,body) for name,body in system.bodies.items() if not dynamic_only or body.dynamic]
    signature=tuple((name,id(body),id(body.model),body.model.X.device,body.model.X.dtype,
        body.model.n_node,body.model.n_mode,body.model.n_respod_mode) for name,body in entries)
    cache=getattr(system,'_body_batch_plans',None)
    if cache is None:
        cache={}
        system._body_batch_plans=cache
    key=(dynamic_only,rigid)
    if key in cache and cache[key][0]==signature:
        return cache[key][1]
    buckets={}
    for name,body in entries:
        model=body.model
        shape=() if rigid else (model.n_node,model.n_mode,model.n_respod_mode)
        group_key=(model.X.device,model.X.dtype,*shape)
        buckets.setdefault(group_key,[]).append((name,body))
    groups=tuple(BodyBatch(items) for items in buckets.values())
    cache[key]=(signature,groups)
    return groups
