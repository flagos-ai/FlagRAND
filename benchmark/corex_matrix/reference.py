"""Independent CPU Philox oracle with the established CoreX distribution mapping."""
import numpy as np

from benchmark.hygon_matrix.reference import words


def reference(api, seed, offset, n):
    count = 2*n if api == 'uniform_f64' else n
    raw = words(seed, offset//4*4, ((offset%4+count+3)//4)*4)
    if api == 'raw32':
        return raw[offset%4:offset%4+n].view(np.int32).copy()
    logical = raw.reshape(-1,4)[:,[0,2,1,3]].reshape(-1)[offset%4:offset%4+count]
    if api == 'uniform_f64':
        high=logical[::2].astype(np.uint64)>>np.uint64(5)
        low=logical[1::2].astype(np.uint64)>>np.uint64(6)
        return ((1<<53)-((high<<np.uint64(26))|low)).astype(np.float64)*2**-53
    signed=logical.view(np.int32).astype(np.float32)
    unsigned=np.where(signed<0,signed+np.float32(4294967296.0),signed)
    uniform=np.clip(np.float32(1)-unsigned*np.float32(2**-32),np.float32(2**-23),np.float32(1))
    if api == 'uniform_f32':
        return uniform
    u=uniform.astype(np.float64)
    radius=np.sqrt(-2*np.log(u[::2]));angle=2*np.pi*u[1::2]
    values=np.stack([radius*np.cos(angle),radius*np.sin(angle)],axis=1).reshape(-1)
    if api=='lognormal_f32':values=np.exp(values)
    return values.astype(np.float32)
