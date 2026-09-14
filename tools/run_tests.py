#!/usr/bin/env python3
"""Run FlagRAND accuracy and performance tests with the FlagGems protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform as platform_module
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmark.summary_json import aggregate_delivery_results  # noqa: E402
from tools.operator_registry import load_registry  # noqa: E402

TIMEOUT = -100
STATUS_LABELS = {
    "Passed": "OK",
    "Failed": "FAILED",
    "Timeout": "TIMEOUT",
    "Error": "ERROR",
    "NotFound": "NOTFOUND",
    "Skipped": "SKIPPED",
}
DTYPE_MAP = {
    "int32": "int32",
    "torch.int32": "int32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "float64": "fp64",
    "torch.float64": "fp64",
}
PERFORMANCE_MODULES = {
    "nvidia": "benchmark.run_nvidia",
    "metax": "benchmark.run_muxi",
    "ascend-cann900": "benchmark.run_ascend_910c",
    "iluvatar": "benchmark.run_corex",
    "thead": "benchmark.run_ppu",
    "hygon": "benchmark.run_hygon",
}
PLATFORM_ALIASES = {
    "hygon": "hygon",
    "hcu": "hygon",
    "nvidia": "nvidia",
    "cuda": "nvidia",
    "h20": "nvidia",
    "metax": "metax",
    "muxi": "metax",
    "maca": "metax",
    "ascend": "ascend-cann900",
    "ascend_910c": "ascend-cann900",
    "ascend-cann900": "ascend-cann900",
    "iluvatar": "iluvatar",
    "corex": "iluvatar",
    "ppu": "thead",
    "thead": "thead",
}


@dataclass(frozen=True)
class RunnerConfig:
    output_dir: Path
    platform: str
    profile: str
    dump_output: bool
    timeout: int


def pinfo(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def pwarn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def perror(message: str) -> None:
    print(f"[ERROR] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_ops_from_inventory() -> list[dict[str, Any]]:
    data = load_registry(ROOT / "conf" / "operators.yaml")
    ops = data.get("ops")
    if not isinstance(ops, list):
        raise ValueError("conf/operators.yaml must contain an 'ops' list")
    return ops


def _stage_name(operator: dict[str, Any]) -> str | None:
    stages = operator.get("stages", [])
    if not stages:
        return None
    latest = stages[-1]
    if not isinstance(latest, dict) or not latest:
        raise ValueError(f"{operator.get('id')}: stages must use the FlagGems maturity format")
    return next(iter(latest))


def get_ops_to_test(
    catalog: list[dict[str, Any]],
    *,
    ops_arg: str | None,
    op_list_file: str | None,
    stages_arg: str,
    start: str | None,
) -> list[dict[str, Any]]:
    by_id = {str(operator["id"]): operator for operator in catalog}
    if ops_arg:
        ids = [item.strip().lstrip("_") for item in ops_arg.split(",") if item.strip()]
        return [by_id[operator_id] for operator_id in ids]
    if op_list_file:
        ids = [
            line.strip().lstrip("_")
            for line in Path(op_list_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return [by_id[operator_id] for operator_id in ids]

    stages = [stage.strip() for stage in stages_arg.split(",") if stage.strip()]
    if "all" in stages:
        stages = ["alpha", "beta", "stable"]
    unsupported = set(stages) - {"alpha", "beta", "stable", "removed"}
    if unsupported:
        raise ValueError(f"unsupported stage names: {sorted(unsupported)}")
    return [
        operator
        for operator in catalog
        if _stage_name(operator) in stages and (start is None or str(operator["id"]) >= start)
    ]


def detect_platform() -> str:
    configured = os.environ.get("FLAGRAND_PLATFORM")
    if configured:
        name = configured.strip().lower()
        if name not in PLATFORM_ALIASES:
            raise RuntimeError(f"unsupported FlagRAND platform: {name}")
        return PLATFORM_ALIASES[name]

    from flagrand.runtime import current_vendor

    vendor = current_vendor()
    try:
        return PLATFORM_ALIASES[vendor]
    except KeyError as exc:
        raise RuntimeError(f"unsupported FlagRAND vendor: {vendor}") from exc


def supported_operator_ids(selected_platform: str) -> set[str]:
    if selected_platform == "hygon":
        from benchmark.hygon_matrix.config import ALL_DISTRIBUTIONS

        return set(ALL_DISTRIBUTIONS)
    if selected_platform == "nvidia":
        from benchmark.nvidia_matrix.config import ALL_DISTRIBUTIONS

        return set(ALL_DISTRIBUTIONS)
    if selected_platform == "ascend-cann900":
        from benchmark.ascend_910c.config import ASCEND_910C_SUPPORTED_DISTRIBUTIONS

        return set(ASCEND_910C_SUPPORTED_DISTRIBUTIONS)
    if selected_platform == "metax":
        from benchmark.muxi_matrix.config import ALL_DISTRIBUTIONS

        return set(ALL_DISTRIBUTIONS)
    if selected_platform == "iluvatar":
        from benchmark.corex_matrix.config import ALL_DISTRIBUTIONS

        return set(ALL_DISTRIBUTIONS)
    if selected_platform == "thead":
        from benchmark.muxi_matrix.config import ALL_DISTRIBUTIONS

        return set(ALL_DISTRIBUTIONS)
    raise RuntimeError(f"unsupported FlagRAND platform: {selected_platform}")


def probe_env(selected_platform: str) -> dict[str, Any]:
    import torch
    import triton

    import flagrand
    from flagrand.runtime import current_vendor, device

    device_count = device.device_count
    device_name = device.torch_device_fn.get_device_name(0) if device_count else "N/A"

    try:
        flagtree_version = metadata.version("flagtree")
    except metadata.PackageNotFoundError:
        flagtree_version = None
    os_release = platform_module.freedesktop_os_release()
    torch_info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": device_name,
        "device_count": device_count,
    }
    if selected_platform == "nvidia":
        properties = torch.cuda.get_device_properties(0)
        torch_info.update(
            cuda_runtime=torch.version.cuda,
            compute_capability=f"{properties.major}.{properties.minor}",
            total_memory=properties.total_memory,
        )
    return {
        "architecture": platform_module.machine(),
        "os_name": os_release["ID"],
        "os_release": os_release["VERSION_ID"],
        "python": platform_module.python_version(),
        "torch": torch_info,
        "flagtree": flagtree_version,
        "triton": {
            "version": triton.__version__,
            "has_config": hasattr(triton, "Config"),
        },
        "flagrand": {
            "version": getattr(flagrand, "__version__", "0.1.0"),
            "vendor": current_vendor(),
            "device": device_name,
        },
    }


def worker_env(platform: str, gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    device = str(gpu_id)
    if platform == "ascend-cann900":
        env["ASCEND_RT_VISIBLE_DEVICES"] = device
        env["NPU_VISIBLE_DEVICES"] = device
    elif platform == "hygon":
        visible = env.get("HIP_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES"))
        selected = visible.split(",")[gpu_id] if visible else device
        env["HIP_VISIBLE_DEVICES"] = selected
        env["CUDA_VISIBLE_DEVICES"] = selected
    elif platform == "iluvatar":
        visible = env.get("CUDA_VISIBLE_DEVICES", env.get("ILUVATAR_VISIBLE_DEVICES"))
        selected = visible.split(",")[gpu_id] if visible else device
        env["ILUVATAR_VISIBLE_DEVICES"] = selected
        env["CUDA_VISIBLE_DEVICES"] = selected
    else:
        env["CUDA_VISIBLE_DEVICES"] = device
    python_paths = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_cmd(
    operator_id: str,
    phase: str,
    command: list[str],
    *,
    env: dict[str, str],
    config: RunnerConfig,
) -> int:
    stdout: Any = subprocess.DEVNULL
    stderr: Any = subprocess.DEVNULL
    handles = []
    if config.dump_output:
        operator_dir = config.output_dir / operator_id
        stdout = (operator_dir / f"{phase}_stdout.log").open("w", encoding="utf-8")
        stderr = (operator_dir / f"{phase}_stderr.log").open("w", encoding="utf-8")
        handles = [stdout, stderr]
        stderr.write(f"[CMD] {' '.join(command)}\n[CWD] {ROOT}\n\n")
        stderr.flush()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                               start_new_session=env.get("FLAGRAND_MANAGED_PROCESS_GROUP") != "1")
    try:
        process.wait(timeout=config.timeout)
        return process.returncode
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return TIMEOUT
    finally:
        for handle in handles:
            handle.close()


def parse_accuracy_data(result_file: Path) -> dict[str, Any]:
    raw_data = json.loads(result_file.read_text(encoding="utf-8"))
    passed: list[str] = []
    skipped: dict[str, list[str]] = {}
    failed: dict[str, list[str]] = {}
    for test_case, item in raw_data.items():
        params = [test_case.split("[", 1)[0]]
        params.extend(str(value).replace(" ", "") for value in item.get("params", {}).values())
        identity = ":".join(params)
        result = item.get("result")
        if result == "passed":
            passed.append(identity)
        elif result == "skipped":
            skipped.setdefault(item.get("reason", "Unknown"), []).append(identity)
        else:
            failed.setdefault(item.get("reason", "Unknown"), []).append(identity)
    total = len(passed) + sum(map(len, skipped.values())) + sum(map(len, failed.values()))
    if failed:
        status = "Failed"
    elif passed:
        status = "Passed"
    elif skipped:
        status = "Skipped"
    else:
        status = "NotFound"
    details: dict[str, Any] = {}
    if failed:
        details["failed"] = failed
    if skipped:
        details["skipped"] = skipped
    return {
        "total": total,
        "skipped": sum(map(len, skipped.values())),
        "failed": sum(map(len, failed.values())),
        "passed": len(passed),
        "status": status,
        "details": details,
    }


def _accuracy_test_paths(operator: dict[str, Any], platform: str) -> list[Path]:
    if platform == "hygon":
        return [ROOT / str(operator["accuracy_test"]), ROOT / "tests/accuracy/test_philox_hygon.py"]
    if platform == "ascend-cann900":
        matches = [path for path in operator["platform_accuracy_tests"] if "ascend_910c" in path]
        return [ROOT / matches[0]]
    paths = [ROOT / str(operator["accuracy_test"])]
    if platform == "nvidia":
        paths.extend(ROOT / str(path) for path in operator["platform_accuracy_tests"] if "nvidia" in str(path))
    elif platform == "metax":
        paths.extend(ROOT / str(path) for path in operator["platform_accuracy_tests"] if "metax" in str(path))
    return paths


def run_accuracy(gpu_id: int, operator: dict[str, Any], config: RunnerConfig) -> dict[str, Any]:
    operator_id = str(operator["id"])
    operator_dir = config.output_dir / operator_id
    ensure_dir(operator_dir)
    raw_file = operator_dir / ".accuracy_raw.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(str(path) for path in _accuracy_test_paths(operator, config.platform)),
        "-m",
        operator_id,
        "--record",
        "json",
        "--output",
        str(raw_file),
        "--ref",
        "cpu",
        "--continue-on-collection-errors",
        "-vs",
    ]
    started = time.monotonic()
    code = run_cmd(operator_id, "accuracy", command, env=worker_env(config.platform, gpu_id), config=config)
    duration = time.monotonic() - started
    if code == TIMEOUT:
        return {
            "status": "Timeout",
            "exit_code": TIMEOUT,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": duration,
        }
    if not raw_file.exists():
        return {
            "status": "Error",
            "exit_code": code,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 1,
            "duration": duration,
            "data_file": None,
        }
    result_file = operator_dir / "accuracy_result.json"
    raw_file.replace(result_file)
    result = parse_accuracy_data(result_file)
    result.update(
        {
            "exit_code": code,
            "duration": duration,
            "data_file": str(result_file.relative_to(config.output_dir)),
        }
    )
    return result


def _performance_command(
    gpu_id: int,
    operator_id: str,
    raw_dir: Path,
    config: RunnerConfig,
) -> list[str]:
    module = os.environ.get("FLAGRAND_PERF_MODULE", PERFORMANCE_MODULES[config.platform])
    profile = config.profile
    benchmark_profile = (
        "matrix"
        if config.platform in {"iluvatar", "nvidia", "hygon"} and profile == "acceptance"
        else ("full" if profile == "acceptance" else profile)
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--profile",
        benchmark_profile,
        "--distributions",
        operator_id,
        "--results-dir",
        str(raw_dir),
    ]
    if config.platform == "ascend-cann900":
        command.extend(("--device", "0", "--allow-non-cann9", "--allow-unverified-910c"))
        if profile == "acceptance":
            command.append("--acceptance-profile")
    elif config.platform == "thead":
        command.append("--skip-torch-reference")
        if profile == "acceptance":
            command.append("--acceptance-profile")
    elif config.platform == "metax":
        command.extend(("--skip-torch-reference", "--with-mcrand"))
        if profile == "acceptance":
            command.extend(
                (
                    "--acceptance-profile",
                    "--acceptance-baseline",
                    "mcrand",
                    "--strict-acceptance",
                )
            )
    return command


def _read_benchmark_records(raw_dir: Path) -> list[dict[str, Any]]:
    record_files = list(raw_dir.glob("*/records.jsonl"))
    if len(record_files) != 1:
        raise RuntimeError(f"expected one records.jsonl under {raw_dir}, found {len(record_files)}")
    return [json.loads(line) for line in record_files[0].read_text(encoding="utf-8").splitlines() if line]


def normalise_performance(operator_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    flagrand_records = [
        record
        for record in records
        if record.get("backend") == "flagrand_public" and record.get("distribution") == operator_id
    ]
    details_by_dtype: dict[str, list[dict[str, Any]]] = {}
    failed = []
    missing_baselines: dict[str, list[str]] = {}
    baseline_name = next(
        (
            str(record["backend"])
            for record in records
            if record.get("backend") != "flagrand_public" and record.get("distribution") == operator_id
        ),
        "baseline",
    )
    for record in flagrand_records:
        dtype = str(record.get("dtype", "unknown"))
        dtype_details = details_by_dtype.setdefault(dtype, [])
        if record.get("status") != "pass":
            failed.append(record.get("error") or record.get("validation") or record.get("status"))
            continue
        latency = float(record["total_ms_median"])
        speedup = None
        for key in (
            "speedup_vs_hygon_hiprand",
            "speedup_vs_nvidia_curand",
            "speedup_vs_corex_curand",
            "speedup_vs_mcrand",
            "speedup_vs_ppu_acrand",
            "speedup_vs_ppu_curand",
            "speedup_vs_cann_aclnn",
        ):
            if record.get(key) is not None:
                speedup = float(record[key])
                break
        if speedup is None:
            baseline = next(
                (
                    candidate
                    for candidate in records
                    if candidate.get("backend") != "flagrand_public"
                    and candidate.get("distribution") == operator_id
                    and candidate.get("n") == record.get("n")
                ),
                None,
            )
            if baseline is None or baseline.get("status") != "pass":
                reason = (
                    "baseline not run"
                    if baseline is None
                    else str(baseline.get("reason") or baseline.get("error") or baseline.get("status"))
                )
                missing_baselines.setdefault(reason, []).append(f"{operator_id}:({int(record['n'])},)")
                continue
            speedup = float(baseline["total_ms_median"]) / latency
        base_latency = latency * speedup
        dtype_details.append(
            {
                "shape_detail": f"({int(record['n'])},)",
                "latency_base": base_latency,
                "latency": latency,
                "speedup": 0.0 if speedup is None else speedup,
            }
        )
    raw_details = [{"dtype": dtype, "result": values} for dtype, values in sorted(details_by_dtype.items())]
    if failed:
        result = "failed"
        reason = json.dumps(failed, ensure_ascii=False)
    elif missing_baselines:
        result = "skipped"
        details = "; ".join(f"{message} ({len(cases)} shapes)" for message, cases in missing_baselines.items())
        reason = f"{baseline_name}: {details}"
    elif raw_details:
        result = "passed"
        reason = None
    else:
        result = "skipped"
        reason = "No FlagRAND benchmark records were produced."
    payload: dict[str, Any] = {
        "result": result,
        "test_case": f"benchmark/test_random_perf.py::{operator_id}",
        "details": raw_details,
    }
    if reason:
        payload["reason"] = reason
    return {operator_id: payload}


def parse_performance_data(operator_id: str, result_file: Path) -> dict[str, Any]:
    raw_data = json.loads(result_file.read_text(encoding="utf-8"))
    operator_data = raw_data[operator_id]
    result = str(operator_data.get("result", "NotFound"))
    benchmark_data: dict[str, Any] = {}
    for item in operator_data.get("details", []):
        dtype = DTYPE_MAP.get(str(item["dtype"]), str(item["dtype"]))
        details: dict[str, Any] = {}
        speedups = []
        for record in item.get("result", []):
            shape = str(record.get("shape_detail", "Unknown")).replace(" ", "")
            speedup = float(record.get("speedup", 0.0))
            details[shape] = {
                "base": float(record.get("latency_base", 0.0)),
                "gems": float(record.get("latency", 0.0)),
                "speedup": speedup,
            }
            speedups.append(speedup)
        benchmark_data[dtype] = {
            "result": "OK" if details else "Unknown",
            "details": details,
            "speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        }
    return {
        "status": result.title(),
        "data": benchmark_data,
        "test_case": operator_data.get("test_case", "Unknown"),
        **({"reason": operator_data["reason"]} if "reason" in operator_data else {}),
    }


def run_performance(gpu_id: int, operator: dict[str, Any], config: RunnerConfig) -> dict[str, Any]:
    operator_id = str(operator["id"])
    operator_dir = config.output_dir / operator_id
    raw_dir = operator_dir / "performance_raw"
    ensure_dir(raw_dir)
    command = _performance_command(gpu_id, operator_id, raw_dir, config)
    started = time.monotonic()
    code = run_cmd(operator_id, "performance", command, env=worker_env(config.platform, gpu_id), config=config)
    duration = time.monotonic() - started
    if code == TIMEOUT:
        return {"status": "Timeout", "exit_code": TIMEOUT, "duration": duration, "data": {}}
    try:
        records = _read_benchmark_records(raw_dir)
    except RuntimeError:
        return {"status": "Error", "exit_code": code, "duration": duration, "data": {}}
    result_file = operator_dir / "performance_result.json"
    result_file.write_text(
        json.dumps(normalise_performance(operator_id, records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = {
        "duration": duration,
        "exit_code": code,
        "data_file": str(result_file.relative_to(config.output_dir)),
        "data": {},
    }
    result.update(parse_performance_data(operator_id, result_file))
    if code != 0:
        if result["status"] != "Failed":
            result["reason"] = f"benchmark command exited with {code}"
        result["status"] = "Failed"
    return result


def _worker(
    gpu_id: int,
    work_queue: Queue,
    event_queue: Queue,
    config: RunnerConfig,
) -> None:
    worker_result = {}
    try:
        while True:
            operator = work_queue.get()
            if operator is None:
                break
            operator_id = str(operator["id"])
            event_queue.put(("start", gpu_id, "accuracy", operator_id))
            accuracy = run_accuracy(gpu_id, operator, config)
            event_queue.put(("done", gpu_id, "accuracy", operator_id, accuracy["status"], accuracy["duration"]))
            event_queue.put(("start", gpu_id, "performance", operator_id))
            performance = run_performance(gpu_id, operator, config)
            event_queue.put(
                ("done", gpu_id, "performance", operator_id, performance["status"], performance["duration"])
            )
            worker_result[operator_id] = {
                "customized": True,
                "accuracy": accuracy,
                "performance": performance,
            }
            (config.output_dir / f"summary{gpu_id}.json").write_text(
                json.dumps(worker_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    finally:
        event_queue.put(("exit", gpu_id))


def _display_events(event_queue: Queue, worker_count: int, operator_count: int) -> None:
    exited = 0
    completed = 0
    total = operator_count * 2
    while exited < worker_count:
        event = event_queue.get()
        if event[0] == "exit":
            exited += 1
            continue
        if event[0] == "start":
            _, gpu_id, phase, operator_id = event
            pinfo(f"[GPU {gpu_id}] {phase:<11} {operator_id} ...")
            continue
        _, gpu_id, phase, operator_id, status, duration = event
        completed += 1
        pinfo(
            f"[GPU {gpu_id}] {phase:<11} {operator_id:<15} "
            f"[{STATUS_LABELS.get(status, status.upper()):<8} {duration:>6.1f}s] "
            f"({completed}/{total})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", help="Comma-separated operator IDs")
    parser.add_argument("--op-list-file", help="File containing one operator ID per line")
    parser.add_argument("--start", help="First operator ID, in lexicographic order")
    parser.add_argument("--gpus", default="0", help="Comma-separated device IDs or 'all'")
    parser.add_argument("--output-dir", help="Result directory; defaults to logs_results_<timestamp>")
    parser.add_argument("--stages", default="stable", help="alpha,beta,stable,removed,or all")
    parser.add_argument("--dump-output", action="store_true", help="Save stdout/stderr for every phase")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--timeout", type=int, default=4800, help="Timeout in seconds for each phase")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = dt.datetime.now()
    selected_platform = detect_platform()
    if os.environ.get("FLAGRAND_MANAGED_PROCESS_GROUP") == "1":
        # Release the probe's context before a benchmark starts; the supervisor is CPU-only.
        env_info = json.loads(subprocess.check_output(
            [sys.executable, "-c", "import json,sys; from tools.run_tests import probe_env; print(json.dumps(probe_env(sys.argv[1])))", selected_platform],
            cwd=ROOT, text=True))
    else:
        env_info = probe_env(selected_platform)
    catalog = get_ops_from_inventory()
    operators = get_ops_to_test(
        catalog,
        ops_arg=args.ops,
        op_list_file=args.op_list_file,
        stages_arg=args.stages,
        start=args.start,
    )
    supported = supported_operator_ids(selected_platform)
    unsupported = [str(operator["id"]) for operator in operators if str(operator["id"]) not in supported]
    if unsupported:
        pwarn(f"{selected_platform} does not implement: {', '.join(unsupported)}")
    operators = [operator for operator in operators if str(operator["id"]) in supported]
    if not operators:
        perror("No operators selected.")
        return 1
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"logs_results_{started_at:%Y%m%d_%H%M}")
    ensure_dir(output_dir)

    if args.gpus.strip().lower() == "all":
        gpu_ids = list(range(int(env_info["torch"]["device_count"])))
    else:
        gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    gpu_ids = gpu_ids[: len(operators)]
    if not gpu_ids:
        perror("No devices selected.")
        return 1

    profile = os.environ.get("FLAGRAND_PERF_PROFILE", "smoke")
    config = RunnerConfig(
        output_dir=output_dir.resolve(),
        platform=selected_platform,
        profile=profile,
        dump_output=args.dump_output,
        timeout=args.timeout,
    )
    pinfo(f"Platform ... {selected_platform}")
    pinfo(f"Testing {len(operators)} operators on devices {gpu_ids} ...")
    pinfo(f"Results ... {config.output_dir}")

    work_queue: Queue = Queue()
    event_queue: Queue = Queue()
    for operator in operators:
        work_queue.put(operator)
    workers = [Process(target=_worker, args=(gpu_id, work_queue, event_queue, config)) for gpu_id in gpu_ids]
    for _ in workers:
        work_queue.put(None)
    for worker in workers:
        worker.start()
    _display_events(event_queue, len(workers), len(operators))
    for worker in workers:
        worker.join()

    result: dict[str, Any] = {}
    for gpu_id in gpu_ids:
        summary_file = config.output_dir / f"summary{gpu_id}.json"
        if summary_file.exists():
            result.update(json.loads(summary_file.read_text(encoding="utf-8")))
            summary_file.unlink()

    ended_at = dt.datetime.now()
    duration = ended_at - started_at
    final_data = {
        "timestamp": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration": str(duration).split(".", 1)[0],
        "env": env_info,
        "result": aggregate_delivery_results(result),
    }
    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(final_data, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = sum(
        phase.get("status") in {"Failed", "Error", "Timeout", "NotFound"}
        for operator_result in result.values()
        for phase in (operator_result["accuracy"], operator_result["performance"])
    )
    missing = {str(operator["id"]) for operator in operators} - set(result)
    failed += len(missing) * 2
    if missing:
        perror(f"Missing worker results for: {', '.join(sorted(missing))}")
    pinfo(f"Test completed with {failed} failed phase(s).")
    pinfo(f"Summary ... {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
