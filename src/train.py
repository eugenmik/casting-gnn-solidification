"""
src/train.py

Training / fine-tuning for MeshGraphNetDelta on graph-based heat diffusion.

Model predicts dT_norm per node for the next step.
We train with unroll K steps:
- sample a start index s
- iteratively predict dT over k=0..K-1
- optionally use teacher forcing (mix GT and predicted T for next input)
- compute loss on dT_norm (Huber) with:
    * early-time weighting
    * boundary-weighted loss (boundary_w > 1 => boundary nodes emphasized)

Validation:
- rollout from t=0 without teacher forcing
- compute TK MSE at fractions cfg.val_fracs
- best metric = mean TK MSE across fractions
- save best checkpoint with EMA weights under 'model_state'

IMPORTANT:
- This trainer is optimized for batch_size=1 + gradient accumulation (accum_steps).
- It supports batch_size>1 via list-collate, but is slower.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import GraphSequenceDataset, list_pt_files
from src.model import MeshGraphNetConfig, MeshGraphNetDelta


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    data_dir: str
    weights_dir: str
    run_name: str

    epochs: int = 60
    batch_size: int = 1
    accum_steps: int = 4
    lr: float = 2e-4
    lr_min: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    val_ratio: float = 0.05
    seed: int = 42
    num_workers: int = 4

    time_mode: str = "random"  # random|fixed|mix_early
    fixed_t: int = 0
    val_fracs: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)

    delta_stats_path: str = "data/index/delta_stats.json"

    hidden_dim: int = 128
    num_blocks: int = 15
    mlp_layers: int = 2
    use_checkpoint: bool = True
    huber_beta: float = 1.0

    unroll_k: int = 4
    unroll_gamma: float = 1.0

    # early-time weighting
    early_w0: float = 8.0
    early_w1: float = 3.0
    early_t0: float = 0.125
    early_t1: float = 0.375

    # teacher forcing schedule
    teacher_start: float = 0.7
    teacher_end: float = 0.1
    teacher_decay_epochs: int = 40

    ema_decay: float = 0.999

    auto_resume: bool = True
    reset_optim_on_resume: bool = False
    init_from: str = ""

    # normalization controls for inputs
    t_norm_eps: float = 1e-3
    clip_t_norm: float = 10.0
    clip_dt_norm: float = 10.0

    # safety
    max_bad_batches_per_epoch: int = 20
    disable_amp_on_nan: bool = True

    use_amp: bool = True
    device: str = "cuda"

    # boundary loss weight (>=1)
    boundary_w: float = 1.0


# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = model.state_dict()
        d = self.decay
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state.items()}


# -----------------------------
# Stats / utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_delta_stats(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    # required keys
    for k in ["T_current", "dT", "dt"]:
        if k not in s or "mean" not in s[k] or "std" not in s[k]:
            raise ValueError(f"delta_stats.json missing required section: {k}")
    return s


def norm_scalar(x: torch.Tensor, mean: float, std: float, eps: float) -> torch.Tensor:
    return (x - mean) / max(std, eps)


def clamp(x: torch.Tensor, lim: float) -> torch.Tensor:
    return torch.clamp(x, -lim, lim)


def teacher_prob(cfg: TrainConfig, epoch_idx: int) -> float:
    if cfg.teacher_decay_epochs <= 0:
        return float(cfg.teacher_end)
    t = min(1.0, max(0.0, epoch_idx / float(cfg.teacher_decay_epochs)))
    return float(cfg.teacher_start + (cfg.teacher_end - cfg.teacher_start) * t)


def early_weight(cfg: TrainConfig, frac: float) -> float:
    """
    Piecewise linear:
      frac <= t0: w0
      t0..t1: linear w0->w1
      > t1: 1.0
    """
    if cfg.time_mode != "mix_early":
        return 1.0
    t0, t1 = cfg.early_t0, cfg.early_t1
    if frac <= t0:
        return float(cfg.early_w0)
    if frac <= t1:
        a = (frac - t0) / max(1e-12, (t1 - t0))
        return float(cfg.early_w0 + a * (cfg.early_w1 - cfg.early_w0))
    return 1.0


def split_train_val(files: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    nv = max(1, int(round(n * val_ratio)))
    val = files[:nv]
    train = files[nv:]
    return train, val


def build_model(cfg: TrainConfig, marker_dim: int) -> MeshGraphNetDelta:
    node_in_dim = 3 + marker_dim + 1 + 1  # pos_norm + markers + T + dt
    edge_in_dim = 4                        # dpos(3) + dist(1)

    mcfg = MeshGraphNetConfig(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=cfg.hidden_dim,
        num_processor_blocks=cfg.num_blocks,
        mlp_layers=cfg.mlp_layers,
        out_dim=1,
        use_checkpoint=cfg.use_checkpoint,
    )
    print(f"[model] building MeshGraphNetDelta with cfg={mcfg}")
    return MeshGraphNetDelta(mcfg)


def _load_state_any(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # checkpoint dict
        for key in ["model_state", "model_state_raw", "state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # raw state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


# -----------------------------
# Feature builder
# -----------------------------
def build_node_features(
    item: Dict[str, Any],
    T_current: torch.Tensor,
    dt: float,
    stats: Dict[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns x: [N, 3+M+1+1]
    """
    pos_norm = item["pos_norm"].to(device)
    markers = item["markers_oh"].to(device)

    Tm = stats["T_current"]["mean"]
    Ts = stats["T_current"]["std"]
    dTm = stats["dT"]["mean"]
    dTs = stats["dT"]["std"]
    dtm = stats["dt"]["mean"]
    dts = stats["dt"]["std"]

    # Normalize T_current globally
    Tn = norm_scalar(T_current, Tm, Ts, cfg.t_norm_eps)
    Tn = clamp(Tn, cfg.clip_t_norm).unsqueeze(-1)  # [N,1]

    # Normalize dt globally
    dt_t = torch.full((pos_norm.size(0), 1), float(dt), device=device, dtype=torch.float32)
    dtn = norm_scalar(dt_t, dtm, dts, cfg.t_norm_eps)
    dtn = clamp(dtn, cfg.clip_dt_norm)             # [N,1]

    x = torch.cat([pos_norm, markers, Tn, dtn], dim=1)
    return x


def huber_loss(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="none")


# -----------------------------
# Rollout (train/val)
# -----------------------------
@torch.no_grad()
def rollout_predict_T(
    model: nn.Module,
    item: Dict[str, Any],
    stats: Dict[str, Any],
    cfg: TrainConfig,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Rollout from t=0 using model predictions only (no teacher forcing).
    Returns T_pred: [S, N]
    """
    temps = item["temps"].to(device)
    times = item["times"].to(device)

    S, N = temps.shape
    if max_steps is not None:
        S = min(S, int(max_steps))
        temps = temps[:S]
        times = times[:S]

    edge_index = item["edge_index"].to(device)
    edge_attr = item["edge_attr"].to(device)

    dTm = stats["dT"]["mean"]
    dTs = stats["dT"]["std"]

    T_pred = torch.empty((S, N), device=device, dtype=torch.float32)
    T_pred[0] = temps[0]

    Tcur = temps[0].clone()
    for t in range(S - 1):
        dt = float(times[t + 1] - times[t])
        x = build_node_features(item, Tcur, dt, stats, cfg, device)
        dT_norm = model(x, edge_index=edge_index, edge_attr=edge_attr)  # [N]
        dT = dT_norm * float(dTs) + float(dTm)
        Tcur = Tcur + dT
        T_pred[t + 1] = Tcur

    return T_pred


def evaluate_val_metrics(
    model_eval: nn.Module,
    val_items: List[Dict[str, Any]],
    stats: Dict[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[float, Dict[float, float]]:
    """
    Returns:
      mean_mse across fractions,
      dict frac->mse
    """
    frac_mse_sum = {f: 0.0 for f in cfg.val_fracs}
    count = 0

    for item in val_items:
        temps = item["temps"]
        S = int(temps.shape[0])
        # precompute indices for fractions
        idxs = {}
        for f in cfg.val_fracs:
            j = int(round(f * (S - 1)))
            j = max(0, min(S - 1, j))
            idxs[f] = j

        T_pred = rollout_predict_T(model_eval, item, stats, cfg, device)
        T_gt = temps.to(device)

        for f, j in idxs.items():
            mse = torch.mean((T_pred[j] - T_gt[j]) ** 2).item()
            frac_mse_sum[f] += mse

        count += 1

    frac_mse = {f: (frac_mse_sum[f] / max(1, count)) for f in cfg.val_fracs}
    mean_mse = sum(frac_mse.values()) / max(1, len(frac_mse))
    return mean_mse, frac_mse


def baseline_val_dT0(val_items: List[Dict[str, Any]], cfg: TrainConfig, device: torch.device) -> Dict[float, float]:
    """
    Baseline: dT=0 => T stays constant at t=0.
    Returns frac->mse
    """
    frac_mse_sum = {f: 0.0 for f in cfg.val_fracs}
    count = 0
    for item in val_items:
        temps = item["temps"].to(device)  # [S,N]
        S = temps.shape[0]
        T0 = temps[0]
        for f in cfg.val_fracs:
            j = int(round(float(f) * (S - 1)))
            j = max(0, min(S - 1, j))
            mse = torch.mean((T0 - temps[j]) ** 2).item()
            frac_mse_sum[f] += mse
        count += 1
    return {f: frac_mse_sum[f] / max(1, count) for f in cfg.val_fracs}


# -----------------------------
# Training
# -----------------------------
def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.weights_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    stats = load_delta_stats(cfg.delta_stats_path)

    files = list_pt_files(cfg.data_dir)
    train_files, val_files = split_train_val(files, cfg.val_ratio, cfg.seed)

    ds_train = GraphSequenceDataset(cfg.data_dir, files=train_files)
    ds_val = GraphSequenceDataset(cfg.data_dir, files=val_files)

    # list-collate to avoid stacking variable-size tensors
    def collate_list(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_list,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_list,
    )

    model = build_model(cfg, marker_dim=ds_train.meta.marker_dim).to(device)
    ema = EMA(model, decay=cfg.ema_decay)

    # init / resume
    resume_path = os.path.join(cfg.weights_dir, f"{cfg.run_name}_resume.pt")
    best_path = os.path.join(cfg.weights_dir, f"{cfg.run_name}_best.pt")

    start_epoch = 1
    best_metric = float("inf")

    if cfg.init_from:
        sd = _load_state_any(cfg.init_from)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded model from: {cfg.init_from}")
        print(f"[init] missing keys: {len(missing)}")
        print(f"[init] unexpected keys: {len(unexpected)}")
        ema = EMA(model, decay=cfg.ema_decay)  # reset EMA to new weights

    if cfg.auto_resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        sd = ckpt.get("model_state_raw", ckpt.get("model_state", None))
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)
        if "ema_state" in ckpt and isinstance(ckpt["ema_state"], dict):
            ema.load_state_dict(ckpt["ema_state"])
        if not cfg.reset_optim_on_resume:
            # optimizer/scaler loaded below after creation
            pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"[resume] loaded checkpoint: {resume_path}")
        print(f"[resume] start_epoch={start_epoch}, best_metric={best_metric}")
        print(f"[resume] reset_optim={cfg.reset_optim_on_resume} (fine-tune)" if cfg.reset_optim_on_resume else "[resume] keep optimizer state")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # cosine schedule to lr_min
    def lr_at_epoch(ep_idx: int) -> float:
        # ep_idx in [1..epochs]
        t = (ep_idx - 1) / max(1, (cfg.epochs - 1))
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * t))

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # If resuming and not resetting optimizer, restore optimizer/scaler
    if cfg.auto_resume and os.path.exists(resume_path) and not cfg.reset_optim_on_resume:
        ckpt = torch.load(resume_path, map_location="cpu")
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

    # Baseline validation
    val_cache = [b[0] for b in dl_val]  # materialize small val set once
    base = baseline_val_dT0(val_cache, cfg, device)
    print("[baseline val] dT=0 => evaluate TK error")
    for f in cfg.val_fracs:
        print(f"  val_frac_{f:0.2f}: TK_mse0={base[f]:.6e} fallback=0.00%")

    bad_batches = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        model.train()

        # update lr
        lr_now = lr_at_epoch(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        p_teacher = teacher_prob(cfg, epoch_idx=epoch - 1)

        running_loss = 0.0
        n_steps = 0

        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(dl_train, start=1):
            # batch is a list of items (dicts)
            for item in batch:
                temps = item["temps"].to(device)  # [S,N]
                times = item["times"].to(device)  # [S]
                S, N = temps.shape
                K = int(cfg.unroll_k)

                if S < (K + 1):
                    continue

                # choose start index s
                if cfg.time_mode == "fixed":
                    s = max(0, min(int(cfg.fixed_t), S - K - 1))
                else:
                    # random start
                    s = random.randint(0, S - K - 1)

                    # mix_early bias: resample from early portion more often
                    if cfg.time_mode == "mix_early":
                        if random.random() < 0.60:
                            s_max = max(0, int(round(cfg.early_t1 * (S - 1))) - K)
                            s = random.randint(0, max(0, s_max))

                edge_index = item["edge_index"].to(device)
                edge_attr = item["edge_attr"].to(device)

                boundary_mask = item["boundary_mask"].to(device)
                if cfg.boundary_w > 1.0:
                    node_w = torch.ones((N,), device=device, dtype=torch.float32)
                    node_w = node_w + (cfg.boundary_w - 1.0) * boundary_mask.to(torch.float32)
                else:
                    node_w = None

                dTm = stats["dT"]["mean"]
                dTs = stats["dT"]["std"]

                # init
                Tcur = temps[s].clone()
                total_loss = 0.0

                # unroll
                for k in range(K):
                    frac = float(s + k) / max(1.0, float(S - 1))
                    w_time = early_weight(cfg, frac) * (cfg.unroll_gamma ** k)

                    dt = float(times[s + k + 1] - times[s + k])
                    x = build_node_features(item, Tcur, dt, stats, cfg, device)

                    # GT normalized dT
                    dT_gt = temps[s + k + 1] - temps[s + k]
                    dT_gt_norm = (dT_gt - float(dTm)) / max(float(dTs), cfg.t_norm_eps)

                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        dT_pred_norm = model(x, edge_index=edge_index, edge_attr=edge_attr)  # [N]
                        per_node = huber_loss(dT_pred_norm, dT_gt_norm, beta=cfg.huber_beta)  # [N]

                        if node_w is not None:
                            per_node = per_node * node_w

                        loss_k = per_node.mean() * float(w_time)

                    total_loss = total_loss + loss_k

                    # update Tcur with teacher forcing
                    dT_pred = dT_pred_norm * float(dTs) + float(dTm)
                    T_next_pred = Tcur + dT_pred
                    if random.random() < p_teacher:
                        Tcur = temps[s + k + 1].clone()
                    else:
                        Tcur = T_next_pred

                # backward (with accumulation)
                total_loss = total_loss / float(cfg.accum_steps)

                if not torch.isfinite(total_loss.detach()):
                    bad_batches += 1
                    if cfg.disable_amp_on_nan and scaler is not None:
                        print("[warn] NaN/Inf detected -> disabling AMP for stability")
                        scaler = None
                    if bad_batches > cfg.max_bad_batches_per_epoch:
                        raise RuntimeError("Too many bad batches in one epoch. Check stats/normalization/data.")
                    continue

                if scaler is None:
                    total_loss.backward()
                else:
                    scaler.scale(total_loss).backward()

                running_loss += float(total_loss.detach().item())
                n_steps += 1

            # optimizer step per accum_steps iterations (approx)
            if (it % cfg.accum_steps) == 0:
                if scaler is None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                else:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()

                opt.zero_grad(set_to_none=True)
                ema.update(model)

        # final step if leftover grads
        if n_steps > 0:
            if scaler is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            else:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)

        # Validation with EMA weights
        model.eval()
        model_eval = build_model(cfg, marker_dim=ds_train.meta.marker_dim).to(device)
        model_eval.load_state_dict(ema.state_dict(), strict=False)
        model_eval.eval()

        val_mean_mse, frac_mse = evaluate_val_metrics(model_eval, val_cache, stats, cfg, device)
        val_frac0 = frac_mse.get(cfg.val_fracs[0], float("nan"))

        dt_epoch = time.time() - t0
        train_loss = (running_loss / max(1, n_steps)) if n_steps > 0 else float("nan")
        print(
            f"[epoch {epoch:04d}] train_loss={train_loss:.6e} "
            f"lr={lr_now:.2e} ema={cfg.ema_decay:.4f} p_teacher={p_teacher:.3f} "
            f"val_TK_mse_mean={val_mean_mse:.6e} val_TK_mse_frac0={val_frac0:.6e} "
            f"time={dt_epoch:.1f}s"
        )
        for f in cfg.val_fracs:
            print(f"  val_frac_{f:0.2f}: TK_mse={frac_mse[f]:.6e} fallback=0.00%")

        # Save resume checkpoint
        resume_payload = {
            "epoch": epoch,
            "best_metric": best_metric,
            "cfg": asdict(cfg),
            "model_state_raw": model.state_dict(),
            "ema_state": ema.state_dict(),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
        }
        _save_checkpoint(resume_path, resume_payload)

        # Save best checkpoint (EMA as model_state for inference)
        if val_mean_mse < best_metric:
            best_metric = val_mean_mse
            best_payload = {
                "epoch": epoch,
                "best_metric": best_metric,
                "cfg": asdict(cfg),
                # For inference tools: prefer EMA
                "model_state": ema.state_dict(),
                "model_state_raw": model.state_dict(),
            }
            _save_checkpoint(best_path, best_payload)
            print(f"[save] best -> {best_path} (best_metric={best_metric:.6f})")

    print(f"[done] best checkpoint: {best_path} (best_metric={best_metric})")
