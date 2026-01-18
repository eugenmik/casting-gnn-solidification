# main.py
from __future__ import annotations

import argparse
from dataclasses import asdict

from src.train import TrainConfig, train


def _parse_fracs(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return (0.0, 0.25, 0.5, 0.75)
    return tuple(float(p) for p in parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Casting Heat Diffusion MVP (MGN dT Unroll)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train MeshGraphNet dT solver (unroll)")

    # IO
    t.add_argument("--data_dir", type=str, default="data_pyg")
    t.add_argument("--weights_dir", type=str, default="weights")
    t.add_argument("--run_name", type=str, default="mgn_dT_unroll_v3_2")

    # Train
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch_size", type=int, default=1)
    t.add_argument("--accum_steps", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--lr_min", type=float, default=2e-5)
    t.add_argument("--weight_decay", type=float, default=1e-4)
    t.add_argument("--grad_clip", type=float, default=1.0)

    # Split / workers
    t.add_argument("--val_ratio", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--num_workers", type=int, default=4)

    # Time sampling
    t.add_argument("--time_mode", type=str, choices=["random", "fixed", "frac", "mix_early"], default="mix_early")
    t.add_argument("--fixed_t", type=int, default=0)
    t.add_argument("--val_fracs", type=str, default="0.0,0.25,0.5,0.75")

    # Stats path
    t.add_argument("--delta_stats_path", type=str, default="data/index/delta_stats.json")

    # Model
    t.add_argument("--hidden_dim", type=int, default=128)
    t.add_argument("--num_blocks", type=int, default=15)
    t.add_argument("--mlp_layers", type=int, default=2)

    # Checkpointing inside model
    t.add_argument("--use_checkpoint", action="store_true", default=True)
    t.add_argument("--no_checkpoint", action="store_true", default=False)

    # Loss / unroll
    t.add_argument("--huber_beta", type=float, default=1.0)
    t.add_argument("--unroll_k", type=int, default=4)
    t.add_argument("--unroll_gamma", type=float, default=1.0)

    # Early-stage weighting
    t.add_argument("--early_w0", type=float, default=8.0)
    t.add_argument("--early_w1", type=float, default=3.0)
    t.add_argument("--early_t0", type=float, default=0.125)
    t.add_argument("--early_t1", type=float, default=0.375)

    # Teacher forcing schedule
    t.add_argument("--teacher_start", type=float, default=0.70)
    t.add_argument("--teacher_end", type=float, default=0.10)
    t.add_argument("--teacher_decay_epochs", type=int, default=40)

    # EMA / resume behavior
    t.add_argument("--ema_decay", type=float, default=0.999)

    t.add_argument("--auto_resume", action="store_true", default=True)
    t.add_argument("--no_auto_resume", action="store_true", default=False)

    t.add_argument("--reset_optim_on_resume", action="store_true", default=True)
    t.add_argument("--no_reset_optim_on_resume", action="store_true", default=False)

    # AMP
    t.add_argument("--use_amp", action="store_true", default=True)
    t.add_argument("--no_amp", action="store_true", default=False)

    # Device
    t.add_argument("--device", type=str, default="cuda")

    # v3.2 additions
    t.add_argument(
        "--boundary_w",
        type=float,
        default=4.0,
        help="Extra node loss weight for boundary nodes (markers_oh>0). Total w = 1 + boundary_w*is_boundary.",
    )
    t.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Initialize model weights from a checkpoint (.pt). Used for fine-tune.",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        val_fracs = _parse_fracs(args.val_fracs)

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
            val_fracs=val_fracs,
            delta_stats_path=args.delta_stats_path,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            mlp_layers=args.mlp_layers,
            use_checkpoint=(False if args.no_checkpoint else True),
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
            auto_resume=(False if args.no_auto_resume else True),
            reset_optim_on_resume=(False if args.no_reset_optim_on_resume else True),
            use_amp=(False if args.no_amp else True),
            device=args.device,
            boundary_w=args.boundary_w,
            init_from=args.init_from,
        )

        print("[config]", asdict(cfg))
        train(cfg)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
