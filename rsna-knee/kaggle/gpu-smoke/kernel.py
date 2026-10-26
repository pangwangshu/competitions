"""Smallest GPU session: prints immediately, checks the GPUs, exits within seconds."""
import json
import subprocess
import time

print("started", time.strftime("%H:%M:%S"), flush=True)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
t = time.time()
print(subprocess.run(["ls", "-R", "/kaggle/input", "-d"], capture_output=True, text=True).stdout[:2000], flush=True)
import glob
print("input dirs", sorted(glob.glob("/kaggle/input/*")), round(time.time() - t, 1), flush=True)
import torch

t = time.time()
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
result = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          "matmul_seconds": None, "cuda_init_seconds": time.time() - t}
t = time.time()
(x @ x).sum().item()
result["matmul_seconds"] = time.time() - t
print(json.dumps(result), flush=True)
open("/kaggle/working/gpu_smoke.json", "w").write(json.dumps(result))
