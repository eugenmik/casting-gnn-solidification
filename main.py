#!/usr/bin/env python3
"""
main.py

Minimal CLI entry point for the Casting Heat Diffusion MVP.

This file intentionally keeps the surface area small:
- Parses CLI arguments
- Builds a TrainConfig
- Calls src.train.train(cfg)

Usage examples:
  python main.py train --data_dir data_pyg --weights_dir weights --run_name run_v1 --epochs 5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Tuple

from src.train import TrainConfig, train


def _tuple_of_floats(text: str) -> Tuple[float, ...]:
    """
    Parse comma-separated floats: "0.0,0.25,0.5,0.75" -> (0.0, 0.25, 0.5, 0.75)
    """
    items = [t.strip() for t in text.split(",") if t.strip()]
    return tuple(float(x) for x in items)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Casting Heat Diffusion MVP (MGN dT Unroll)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # Train subcommand
    # -------------------------
    t = sub.add_parser("train", help="Train / fine-tune the MeshGraphNetDelta model")

    # Paths
    t.add_argument("--data_dir", type=str, required=True, help="Directory with .pt graph files")
    t.add_argument("--weights_dir", type=str, default="weights", help="Where to save checkpoints")
    t.add_argument("--run_name", type=str, required=True, help="Run name prefix for checkpoints/logs")
    t.add_argument("--delta_stats_path", type=str, required=True, help="Path to data/index/delta_stats.json")

    # Training basics
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch_size", type=int, default=1)
    t.add_argument("--accum_steps", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lr_min", type=float, default=2e-5)
    t.add_argument("--weight_decay", type=float, default=1e-4)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--num_workers", type=int, default=4)
    t.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    t.add_argument("--use_amp", action="store_true", help="Enable AMP (mixed precision)")

    # Data split + validation
    t.add_argument("--val_ratio", type=float, default=0.05)
    t.add_argument(
        "--val_fracs",
        type=_tuple_of_floats,
        default=(0.0, 0.25, 0.5, 0.75),
        help="Validation time fractions (comma-separated)",
    )

    # Time sampling modes
    t.add_argument(
        "--time_mode",
        type=str,
        default="random",
        choices=["random", "fixed", "mix_early"],
        help="How to sample time indices for training",
    )
    t.add_argument("--fixed_t", type=int, default=0, help="Used when time_mode=fixed")

    # Model config
    t.add_argument("--hidden_dim", type=int, default=128)
    t.add_argument("--num_blocks", type=int, default=15)
    t.add_argument("--mlp_layers", type=int, default=2)
    t.add_argument("--use_checkpoint", action="store_true", help="Gradient checkpointing inside model blocks")

    # Loss + unroll
    t.add_argument("--huber_beta", type=float, default=1.0)
    t.add_argument("--unroll_k", type=int, default=4, help="Unroll steps for rollout loss")
    t.add_argument("--unroll_gamma", type=float, default=1.0, help="Discount for later rollout steps")

    # Teacher forcing schedule
    t.add_argument("--teacher_start", type=float, default=0.7, help="Teacher forcing probability at epoch=0")
    t.add_argument("--teacher_end", type=float, default=0.1, help="Teacher forcing prob at end of decay")
    t.add_argument("--teacher_decay_epochs", type=int, default=40, help="Epochs to decay teacher forcing")

    # Early-time weighting (for mix_early)
    t.add_argument("--early_w0", type=float, default=8.0)
    t.add_argument("--early_w1", type=float, default=3.0)
    t.add_argument("--early_t0", type=float, default=0.125)
    t.add_argument("--early_t1", type=float, default=0.375)

    # EMA + resume / init
    t.add_argument("--ema_decay", type=float, default=0.999)
    t.add_argument("--init_from", type=str, default="", help="Path to checkpoint/state_dict to initialize from")
    t.add_argument("--auto_resume", action="store_true", help="Auto-resume from *_resume.pt if exists")
    t.add_argument("--no_auto_resume", action="store_true", help="Disable auto-resume explicitly")
    t.add_argument("--reset_optim_on_resume", action="store_true", help="Reset optimizer when resuming/init_from")

    # Stability / NaN handling
    t.add_argument("--t_norm_eps", type=float, default=1e-3)
    t.add_argument("--clip_t_norm", type=float, default=10.0)
    t.add_argument("--clip_dt_norm", type=float, default=10.0)
    t.add_argument("--max_bad_batches_per_epoch", type=int, default=20)
    t.add_argument("--disable_amp_on_nan", action="store_true", help="Turn off AMP if NaNs detected")

    # Boundary weighting
    t.add_argument("--boundary_w", type=float, default=1.0, help="Extra loss weight for boundary nodes (>1 means stronger)")

    return p


def cmd_train(args: argparse.Namespace) -> None:
    auto_resume = bool(args.auto_resume) and not bool(args.no_auto_resume)

    cfg = TrainConfig(
        data_dir=args.data_dir,
        weights_dir=args.weights_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        time_mode=args.time_mode,
        fixed_t=args.fixed_t,
        val_fracs=tuple(args.val_fracs),
        delta_stats_path=args.delta_stats_path,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        mlp_layers=args.mlp_layers,
        use_checkpoint=bool(args.use_checkpoint),
        huber_beta=args.huber_beta,
        unroll_k=args.unroll_k,
        unroll_gamma=args.unroll_gamma,
        early_w0=args.early_w0,
        early_w1=args.early_w1,
        early_t0=args.early_t0,
        early_t1=args.early_t1,
        teacher_start=args.teacher_start,
        teacher_end=args.teacher_end,
        teacher_decay_epochs=args.teacher_decay_epochs,
        ema_decay=args.ema_decay,
        auto_resume=auto_resume,
        reset_optim_on_resume=bool(args.reset_optim_on_resume),
        use_amp=bool(args.use_amp),
        device=args.device,
        boundary_w=args.boundary_w,
        init_from=args.init_from,
        t_norm_eps=args.t_norm_eps,
        clip_t_norm=args.clip_t_norm,
        clip_dt_norm=args.clip_dt_norm,
        max_bad_batches_per_epoch=args.max_bad_batches_per_epoch,
        disable_amp_on_nan=bool(args.disable_amp_on_nan),
    )

    print("[config]", asdict(cfg))
    train(cfg)


def main() -> None:
    p = build_argparser()
    args = p.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()