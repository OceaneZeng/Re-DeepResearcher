#!/usr/bin/env python3
"""
Diagnostic helper for ROCm / PyTorch / accelerate segfaults.
Run this on the target machine: python scripts/diagnose_rocm.py
It will enable faulthandler and print environment variables, try to import torch and run small GPU ops,
inspect bitsandbytes, and attempt to import accelerate within a subprocess so we can observe segfaults.
"""
import os
import sys
import faulthandler
import traceback
import subprocess
import time

faulthandler.enable()

def print_hdr(msg):
    print("\n=== {} ===".format(msg))

print_hdr("ENVIRONMENT")
print("python executable:", sys.executable)
print("python version:", sys.version.replace('\n', ' '))
for k in [
    "HIP_VISIBLE_DEVICES",
    "ROCMLIBS",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ALLOC_CONF",
    "LD_LIBRARY_PATH",
    "PATH",
    "ROCM_PATH",
]:
    print(f"{k}={os.environ.get(k)}")

print_hdr("TORCH IMPORT / ROCM CHECK")
try:
    import torch
    t0 = time.time()
    print("torch.__version__:", getattr(torch, "__version__", None))
    hip = getattr(torch.version, "hip", None)
    print("torch.version.hip:", hip)
    try:
        cuda_avail = torch.cuda.is_available()
    except Exception as e:
        cuda_avail = f"error: {e}"
    print("torch.cuda.is_available():", cuda_avail)
    try:
        devc = torch.cuda.device_count()
        print("torch.cuda.device_count():", devc)
    except Exception as e:
        print("torch.cuda.device_count() raised:", e)

    # test small allocation & device transfer
    try:
        cpu_t = torch.randn(2,2)
        print("cpu allocation ok")
        if hasattr(torch, "device") and torch.cuda.is_available():
            try:
                cuda_t = cpu_t.to("cuda")
                print("moved tensor to cuda ok ->", cuda_t.device)
            except Exception as e:
                print("moving tensor to cuda raised:", repr(e))
    except Exception as e:
        print("small tensor allocation failed:", repr(e))
except Exception as e:
    print("torch import raised/exited with error:", repr(e))
    traceback.print_exc()

print_hdr("BITSANDBYTES CHECK")
try:
    import bitsandbytes as bnb
    print("bitsandbytes imported, version:", getattr(bnb, "__version__", "unknown"))
except Exception as e:
    print("bitsandbytes import failed or not installed:", repr(e))

print_hdr("ACCELERATE IMPORT (subprocess)")
try:
    cmd = [sys.executable, "-c", "import accelerate, sys; print('accelerate', getattr(accelerate, '__version__', 'unknown')); sys.exit(0)"]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("returncode:", proc.returncode)
    print("stdout:\n", proc.stdout)
    print("stderr:\n", proc.stderr)
    if proc.returncode != 0:
        print("Note: non-zero return code from accelerate import. If a segfault occurred you'll typically see 'Segmentation fault' in stderr or the shell.")
except Exception as e:
    print("accelerate subprocess test failed:", repr(e))
    traceback.print_exc()

print_hdr("SUGGESTIONS")
print("If torch import alone fails or causes segfault: reinstall a ROCm-matching PyTorch wheel (do NOT install CUDA wheel).")
print("If accelerate import (subprocess) returns non-zero or segfaults: try in a clean venv and ensure accelerate/version matches environment.")
print("To collect a native backtrace, run under gdb: gdb --args python -c \"import accelerate;\" then run and bt\n")

print('\nDone.')

