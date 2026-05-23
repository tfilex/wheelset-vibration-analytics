import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTION_SRC = os.path.join(PROJECT_ROOT, "src", "prediction")
if PREDICTION_SRC not in sys.path:
    sys.path.append(PREDICTION_SRC)

from model import UniversalHybridRULNet, create_cnn_encoder  # noqa: E402


def build_rul_model(
    temporal_type: str,
    backbone: str,
    in_channels: int,
    hidden_size: int,
    dropout: float,
    freeze_encoder: bool,
) -> torch.nn.Module:
    encoder, enc_dim = create_cnn_encoder(
        backbone_name=backbone,
        in_channels=in_channels,
        pretrained=False,
        freeze=freeze_encoder,
    )
    return UniversalHybridRULNet(
        encoder=encoder,
        encoder_dim=enc_dim,
        temporal_type=temporal_type,
        hidden_size=hidden_size,
        dropout=dropout,
        num_temporal_layers=2,
    )


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile_ms(values_ms: List[float], q: float) -> float:
    return float(np.percentile(np.array(values_ms, dtype=np.float64), q))


def measure_inference_latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    channels: int,
    height: int,
    width: int,
    warmup_runs: int,
    num_runs: int,
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    dummy_input = torch.randn(
        batch_size, seq_len, channels, height, width, device=device
    )

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        synchronize_if_cuda(device)

    run_times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            synchronize_if_cuda(device)
            end = time.perf_counter()
            run_times_ms.append((end - start) * 1000.0)

    avg_ms = float(np.mean(run_times_ms))
    p50_ms = percentile_ms(run_times_ms, 50)
    p95_ms = percentile_ms(run_times_ms, 95)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    print("\n=== Inference Latency Benchmark (RUL temporal models) ===")
    print(f"{'temporal':<14} {'avg_ms':>12} {'p50_ms':>12} {'p95_ms':>12} {'fps':>10}")
    print("-" * 64)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<14} "
            f"{metrics['avg_ms']:>12.2f} "
            f"{metrics['p50_ms']:>12.2f} "
            f"{metrics['p95_ms']:>12.2f} "
            f"{metrics['fps']:>10.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Оценка примерного времени инференса для temporal-архитектур "
            "из prediction/train (lstm, gru, tcn, transformer)."
        )
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument(
        "--temporal-types",
        nargs="+",
        default=["lstm", "gru", "tcn", "transformer"],
        help="Список temporal-моделей для сравнения.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA выбрана, но недоступна. Запустите с --device cpu.")

    device = torch.device(args.device)
    torch.manual_seed(42)

    print(f"[INFO] Device: {device}")
    print(
        "[INFO] Input shape: "
        f"(B={args.batch_size}, S={args.seq_len}, C={args.channels}, H={args.height}, W={args.width})"
    )
    print(f"[INFO] Runs: warmup={args.warmup_runs}, measure={args.num_runs}")

    results: Dict[str, Dict[str, float]] = {}
    for temporal_type in args.temporal_types:
        print(f"\n[INFO] Benchmarking temporal model: {temporal_type}")
        model = build_rul_model(
            temporal_type=temporal_type,
            backbone=args.backbone,
            in_channels=args.channels,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            freeze_encoder=True,
        )
        metrics = measure_inference_latency(
            model=model,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            channels=args.channels,
            height=args.height,
            width=args.width,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
        )
        results[temporal_type] = metrics
        print(
            f"[INFO] {temporal_type}: avg={metrics['avg_ms']:.2f} ms | "
            f"p95={metrics['p95_ms']:.2f} ms | fps={metrics['fps']:.2f}"
        )

    print_results(results)


if __name__ == "__main__":
    main()