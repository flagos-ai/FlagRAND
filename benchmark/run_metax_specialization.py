"""Finite diagnosis of the N=1025 cold-specialization cliff; not formal acceptance."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import traceback


def main():
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/'src')]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--strategy',choices=('original','counter_dynamic'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.strategy is None:
        results={}
        for strategy in ('original','counter_dynamic'):
            env=dict(os.environ,TRITON_CACHE_DIR=str(root/'../../data/cache/metax-specialization'/args.output.name/strategy))
            code=subprocess.call([sys.executable,__file__,'--output',str(args.output),'--strategy',strategy],env=env)
            results[strategy]={'exit_code':code,'report':json.loads((args.output/(strategy+'.json')).read_text())}
            (args.output/'summary.json').write_text(json.dumps(results,indent=2))
        return int(any(r['exit_code'] for r in results.values()))

    import torch
    import triton
    from benchmark.hygon_matrix.reference import words
    from benchmark.muxi_matrix.compat import collect_environment
    from benchmark.muxi_matrix.config import PROFILES
    from benchmark.muxi_matrix.timing import measure
    from flagrand import curand
    from flagrand.metax import philox
    from flagrand.metax.launcher import MetaXKernelLauncher

    if args.strategy=='counter_dynamic':
        for name in ('_RAW_GENERIC_LAUNCHER','_UNIFORM_F32_GENERIC_LAUNCHER'):
            old=getattr(philox,name)
            kernel=triton.jit(old._kernel.fn,do_not_specialize=['base_counter'])
            setattr(philox,name,MetaXKernelLauncher(kernel,constexpr_names=old._constexpr_names))

    report={'environment':collect_environment(),'strategy':args.strategy,'cases':[],
            'scope':'Two APIs, three sizes, default block512/warps4. Fresh disk cache per strategy. '
                    'Same full timing warmup5/repeats11/200ms. CPU profiles are diagnostic, not acceptance timings.',
            'cache_directory':os.environ['TRITON_CACHE_DIR']}

    def save():
        (args.output/(args.strategy+'.json')).write_text(json.dumps(report,indent=2))

    def profile_text(prof,stem):
        text=io.StringIO();pstats.Stats(prof,stream=text).sort_stats('cumtime').print_stats(50)
        (args.output/(stem+'.txt')).write_text(text.getvalue())
        prof.dump_stats(str(args.output/(stem+'.prof')))
        return text.getvalue()

    for api in ('raw32','uniform_f32'):
        op=curand.generate if api=='raw32' else curand.generate_uniform
        dtype=torch.int32 if api=='raw32' else torch.float32
        launcher=getattr(philox,'_RAW_GENERIC_LAUNCHER' if api=='raw32' else '_UNIFORM_F32_GENERIC_LAUNCHER')
        for n in (1023,1025,4097):
            row={'api':api,'n':n};report['cases'].append(row);save()
            stem=f'{args.strategy}-{api}-{n}'
            try:
                out=torch.empty(n,device='cuda',dtype=dtype)
                replay=torch.empty_like(out)
                op(curand.create_generator(seed=1234),out)
                op(curand.create_generator(seed=1234),replay)
                torch.testing.assert_close(out,replay,rtol=0,atol=0)
                gen=curand.create_generator(seed=1234)
                def fn(): return op(gen,out)
                # Observe compilation in the unchanged timing sequence separately.
                prof=cProfile.Profile();prof.enable()
                row['profiled_timing']=measure(fn,PROFILES['full'])
                prof.disable();row['cold_cpu_profile']=profile_text(prof,stem+'-cold')
                row['warm_timing']=measure(fn,PROFILES['full'])
                checks=[]
                for offset in range(4):
                    op(curand.create_generator(seed=1234,offset=offset),out)
                    if api=='raw32':
                        expected=torch.from_numpy(words(1234,offset,n).view('int32').copy())
                        torch.testing.assert_close(out.cpu(),expected,rtol=0,atol=0)
                    else:
                        # Exact configuration equivalence, in addition to the formal distribution tests.
                        op(curand.create_generator(seed=1234,offset=offset),replay,block_size=256,num_warps=4)
                        torch.testing.assert_close(out,replay,rtol=0,atol=0)
                    checks.append({'offset':offset,'status':'pass'})
                row['checks']=checks
                compiled=[]
                for key,kernel in launcher._entries.items():
                    item={'cache_key':repr(key),'n_regs':getattr(kernel,'n_regs',None),'n_spills':getattr(kernel,'n_spills',None),
                          'metadata':str(kernel.metadata),'ir':{}}
                    for kind,value in kernel.asm.items():
                        if isinstance(value,str):
                            digest=hashlib.sha256(value.encode()).hexdigest();filename=digest+'.'+kind
                            (args.output/filename).write_text(value)
                            item['ir'][kind]={'sha256':digest,'file':filename}
                    compiled.append(item)
                row['compiled']=compiled
                prof=cProfile.Profile();prof.enable()
                for _ in range(2000):fn()
                torch.cuda.synchronize();prof.disable()
                row['warm_cpu_profile']=profile_text(prof,stem+'-warm')
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):fn()
                samples=[]
                for _ in range(7):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    samples.append(start.elapsed_time(end)/16)
                row.update(status='pass',graph_ms_samples=samples,graph_ms_median=statistics.median(samples))
            except Exception as exc:
                row.update(status='error',error=repr(exc),traceback=traceback.format_exc())
            save();print(args.strategy,api,n,row['status'],flush=True)
    return int(any(r['status']!='pass' for r in report['cases']))


if __name__=='__main__':
    raise SystemExit(main())
