"""Device and precision configuration shared by reader entry points."""

from contextlib import nullcontext

import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(name=None):
    device = torch.device(name or default_device())
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device mps or cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process; select --device cpu")
    return device


def resolve_amp(device, amp=None, *, training=False):
    if amp is None:
        if device.type != "cuda":
            return "none"
        return "bf16" if training and torch.cuda.is_bf16_supported() else "fp16"
    if amp not in {"bf16", "fp16", "none"}:
        raise ValueError(f"Unsupported precision: {amp}")
    if amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16; select --amp fp16 or none")
    return amp


def autocast_context(device, amp):
    if amp == "none":
        return nullcontext()
    return torch.autocast(device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[amp])
