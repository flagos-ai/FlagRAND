"""Finite CoreX launch diagnosis; this does not replace the fixed native acceptance matrix."""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    import numpy as np
    import torch
    import triton
    from benchmark.corex_matrix.cases import invoke_flagrand, make_flagrand_generator
    from benchmark.corex_matrix.config import Profile
    from benchmark.hygon_matrix.reference import words
    from benchmark.muxi_matrix.timing import measure
    from flagrand import curand
    from flagrand.runtime.backend import get_backend_state
    from flagrand.rng import philox as raw_impl
    from flagrand.corex import philox as corex_impl
    from triton.runtime.driver import driver

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    state = get_backend_state(); original = state.vendor_info
    profile = Profile('diagnostic', (), 5, 5, 20.0)
    report = {'scope': 'CoreX cached launch candidate; short diagnostic timings, not acceptance',
              'environment': {'torch': torch.__version__, 'triton': triton.__version__,
                              'device': torch.cuda.get_device_name(0), 'target': str(driver.active.get_current_target())},
              'correctness': [], 'cases': [], 'profiles': {}, 'compiler': []}

    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))

    def invoke(api, gen, out, block=512, warps=4):
        operation = curand.generate if api == 'raw32' else curand.generate_normal
        return operation(gen, out, block_size=block, num_warps=warps)

    def reference(api, offset, n):
        raw = words(0x1234567887654321, offset // 4 * 4, ((offset % 4 + n + 3) // 4) * 4)
        if api == 'raw32':
            return torch.from_numpy(raw[offset % 4:offset % 4+n].view(np.int32).copy())
        logical = raw.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1)[offset % 4:offset % 4+n]
        signed = logical.view(np.int32).astype(np.float32)
        unsigned = np.where(signed < 0, signed + np.float32(4294967296.0), signed)
        uniform = np.clip(np.float32(1) - unsigned * np.float32(2**-32), np.float32(2**-23), np.float32(1)).astype(np.float64)
        radius = np.sqrt(-2 * np.log(uniform[::2])); angle = 2*np.pi*uniform[1::2]
        return torch.from_numpy(np.stack([radius*np.cos(angle), radius*np.sin(angle)],axis=1).reshape(-1).astype(np.float32))

    save()
    for mode in ('regular', 'cached'):
        state.vendor_info = replace(original, cached_binary_launch=mode == 'cached')
        for api in ('raw32', 'normal_f32'):
            dtype = torch.int32 if api == 'raw32' else torch.float32
            for offset in (0, 4, 2**33-4, 2**34-4):
                n = 1025 if api == 'raw32' else 1026
                row = dict(mode=mode, api=api, offset=offset, n=n)
                try:
                    out = torch.empty(n, device='cuda', dtype=dtype)
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        invoke(api, make_flagrand_generator(0x1234567887654321, offset=offset), out)
                    stream.synchronize()
                    expected = reference(api, offset, n)
                    torch.testing.assert_close(out.cpu(), expected, rtol=0 if api=='raw32' else 2e-5, atol=0 if api=='raw32' else 2e-6)
                    row['status'] = 'pass'
                except Exception:
                    row.update(status='error', traceback=traceback.format_exc())
                report['correctness'].append(row); save()
            out = torch.empty(4096,device='cuda',dtype=dtype)
            gen = make_flagrand_generator(1234)
            for _ in range(16):invoke_flagrand(api,gen,out)
            torch.cuda.synchronize()
            prof = cProfile.Profile(); prof.enable()
            for _ in range(1000):invoke_flagrand(api,gen,out)
            torch.cuda.synchronize();prof.disable()
            stem = args.output / f'{mode}-{api}-cpu'
            prof.dump_stats(str(stem.with_suffix('.prof')))
            text = io.StringIO();pstats.Stats(prof,stream=text).sort_stats('cumtime').print_stats(35)
            stem.with_suffix('.txt').write_text(text.getvalue())
            report['profiles'][f'{mode}-{api}'] = text.getvalue();save()
            configs = ((512,4),) if mode=='regular' else ((512,4),(256,4),(128,2))
            for n in (4096,1048576,16777216):
                for block,warps in configs:
                    row = dict(mode=mode,api=api,n=n,block=block,warps=warps)
                    try:
                        out=torch.empty(n,device='cuda',dtype=dtype)
                        gen=make_flagrand_generator(1234)
                        invoke(api,gen,out,block,warps)
                        state.vendor_info=original
                        expected=torch.empty_like(out)
                        invoke(api,make_flagrand_generator(1234),expected)
                        state.vendor_info=replace(original,cached_binary_launch=mode=='cached')
                        torch.testing.assert_close(out,expected,rtol=0,atol=0)
                        row.update(validation='bitwise equal to original strategy',**measure(lambda:invoke(api,gen,out,block,warps),profile))
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(16):invoke(api,gen,out,block,warps)
                        samples=[]
                        for _ in range(7):
                            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            begin.record();graph.replay();end.record();end.synchronize()
                            samples.append(begin.elapsed_time(end)/16)
                        row.update(status='pass',graph_ms_samples=samples,graph_ms_median=statistics.median(samples),graph_calls=16)
                    except Exception:
                        row.update(status='error',traceback=traceback.format_exc())
                    report['cases'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k in ('mode','api','n','block','warps','status','total_ms_median','graph_ms_median')}),flush=True)
    kernels=(raw_impl._philox_kernel,raw_impl._philox_generic_kernel,corex_impl._normal_aligned_kernel,corex_impl._normal_pair_kernel)
    for kernel in kernels:
        for cache in kernel.cache.values():
            for compiled in cache.values():
                stem=args.output/compiled.hash
                for kind,content in compiled.asm.items():
                    if isinstance(content,str):stem.with_suffix('.'+kind).write_text(content)
                report['compiler'].append(dict(name=compiled.name,hash=compiled.hash,metadata=compiled.metadata._asdict(),
                    n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),formats=list(compiled.asm)))
    state.vendor_info=original
    # Target metadata is a namedtuple and serializes as a tuple via json.
    save()
    return int(any(r['status']!='pass' for r in report['cases']+report['correctness']))


if __name__=='__main__':
    raise SystemExit(main())
