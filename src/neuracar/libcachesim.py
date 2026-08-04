"""Small adapter for the libCacheSim ``cachesim`` CLI used by the baselines."""

import re
import subprocess


def run_libcachesim_baseline(
    executable: str,
    trace_path: str,
    algorithm: str,
    capacity: int,
    trace_params: str = "time-col=1,obj-id-col=2,size-col=3,delimiter=,,has-header=false",
) -> tuple[float, float | None, str]:
    """Run one baseline and return (request hit ratio, byte hit ratio, raw output)."""
    command = [
        executable,
        trace_path,
        "csv",
        algorithm,
        str(capacity),
        "-t",
        trace_params,
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    miss_matches = re.findall(r"(?<!byte )miss ratio\s*[:=]?\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    byte_matches = re.findall(r"byte miss ratio\s*[:=]?\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    if not miss_matches:
        raise RuntimeError(f"could not parse libCacheSim miss ratio from output:\n{output}")
    request_hit_ratio = 1.0 - float(miss_matches[-1])
    byte_hit_ratio = 1.0 - float(byte_matches[-1]) if byte_matches else None
    return request_hit_ratio, byte_hit_ratio, output
