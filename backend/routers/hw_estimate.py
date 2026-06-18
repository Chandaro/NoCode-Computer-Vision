"""Shared heuristics for pre-flight training resource estimates (GPU VRAM +
system RAM). Estimates are approximate — calibrated to typical training
footprints — and meant to catch an out-of-memory config before a run starts."""


def device_status() -> dict:
    """Live GPU / RAM availability."""
    out: dict = {"device": "CPU", "gpu": None, "ram": {}}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["device"] = "GPU"
            out["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_gb": round(total / 1e9, 2),
                "free_gb": round(free / 1e9, 2),
            }
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram"] = {"total_gb": round(vm.total / 1e9, 2),
                      "available_gb": round(vm.available / 1e9, 2)}
    except Exception:
        pass
    return out


def verdict(est_vram: float, free_gb: float, overhead: float, per_eff: float):
    """Return (verdict, suggested_batch). per_eff = VRAM per single sample."""
    if est_vram <= free_gb * 0.85:
        return "fits", None
    if est_vram <= free_gb:
        return "tight", None
    budget = max(0.2, free_gb * 0.8 - overhead)
    return "over", max(1, int(budget / max(per_eff, 1e-6)))


# Classification / Conv-Builder backbones: (overhead_gb, per_image_gb at 224px)
CLS_BACKBONES = {
    "resnet18": (0.5, 0.05), "resnet34": (0.6, 0.08), "resnet50": (0.9, 0.14),
    "mobilenet_v3_small": (0.4, 0.03),
    "efficientnet_b0": (0.6, 0.07), "efficientnet_b1": (0.7, 0.09),
    "convnext_tiny": (0.9, 0.15),
}


def estimate_classification(backbone: str, imgsz: int, batch: int,
                            freeze: bool = False) -> tuple[float, float, float]:
    """VRAM estimate for a torchvision-backbone classifier.
    Returns (est_vram_gb, overhead_gb, per_sample_gb)."""
    overhead, per_img = CLS_BACKBONES.get(backbone, (0.6, 0.08))
    scale   = (max(32, imgsz) / 224.0) ** 2
    fr      = 0.6 if freeze else 1.0          # frozen backbone stores no backbone grads
    per_eff = per_img * scale * fr
    est     = round(overhead + per_eff * max(1, batch), 2)
    return est, overhead, per_eff


def estimate_custom_cnn(input_size: int, batch: int,
                        params_m: float = 1.0) -> tuple[float, float, float]:
    """VRAM estimate for a from-scratch custom CNN. Activation memory scales with
    input area; weights/grads/optimizer scale with parameter count.
    Returns (est_vram_gb, overhead_gb, per_sample_gb)."""
    overhead = round(0.3 + max(0.0, params_m) * 0.012, 3)   # ~12 bytes/param (w+grad+adam)
    per_eff  = 0.012 * (max(32, input_size) / 96.0) ** 2
    est      = round(overhead + per_eff * max(1, batch), 2)
    return est, overhead, per_eff
