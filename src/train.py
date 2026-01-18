# src/train.py
from __future__ import annotations

import os
import json
import math
import time
import glob
import random
import inspect
import importlib
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception as e:
    raise ImportError(
        "torch_geometric не найден. Установи torch-geometric в окружение cast_gnn."
    ) from e

# ---------- AMP compat ----------
def _autocast(enabled: bool, device: torch.device):
    # torch >= 2.0
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        # old style
        return torch.cuda.amp.autocast(enabled=enabled)

def _GradScaler(enabled: bool, device: torch.device):
    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------- Config ----------
@dataclass
class TrainConfig:
    # IO
    data_dir: str = "data_pyg"
    weights_dir: str = "weights"
    run_name: str = "mgn_dT_unroll_v3_2"

    # Train
    epochs: int = 10
    batch_size: int = 1
    accum_steps: int = 4
    lr: float = 5e-5
    lr_min: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Split / workers
    val_ratio: float = 0.05
    seed: int = 42
    num_workers: int = 4

    # Time sampling
    time_mode: str = "mix_early"  # random | fixed | frac | mix_early
    fixed_t: int = 0
    val_fracs: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)

    # Stats
    delta_stats_path: str = "data/index/delta_stats.json"

    # Model
    hidden_dim: int = 128
    num_blocks: int = 15
    mlp_layers: int = 2
    use_checkpoint: bool = True

    # Loss / unroll
    huber_beta: float = 1.0
    unroll_k: int = 4
    unroll_gamma: float = 1.0

    # Early-stage weighting
    early_w0: float = 8.0
    early_w1: float = 3.0
    early_t0: float = 0.125
    early_t1: float = 0.375

    # Teacher forcing schedule (prob use GT for next step)
    teacher_start: float = 0.70
    teacher_end: float = 0.10
    teacher_decay_epochs: int = 40

    # EMA / resume
    ema_decay: float = 0.999
    auto_resume: bool = False
    reset_optim_on_resume: bool = True

    # AMP / device
    use_amp: bool = True
    device: str = "cuda"

    # v3.2
    boundary_w: float = 4.0
    init_from: str = ""  # path to model weights (state_dict or full ckpt)

    # Stability
    t_norm_eps: float = 1e-3
    clip_t_norm: float = 10.0
    clip_dt_norm: float = 10.0
    max_bad_batches_per_epoch: int = 20
    disable_amp_on_nan: bool = True


# ---------- EMA ----------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init(model)

    @torch.no_grad()
    def _init(self, model: nn.Module):
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
            else:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not hasattr(self, "_backup"):
            return
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        delattr(self, "_backup")


# ---------- Utils ----------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _list_pt_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")
    return files

def _split_files(files: List[str], val_ratio: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(files)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(files) * val_ratio)))
    val_idx = set(idx[:n_val])
    train_files = [files[i] for i in idx if i not in val_idx]
    val_files = [files[i] for i in idx if i in val_idx]
    return train_files, val_files

def _load_stats(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        j = json.load(f)
    Tm = float(j["T_current"]["mean"])
    Ts = float(j["T_current"]["std"])
    dTm = float(j["dT"]["mean"])
    dTs = float(j["dT"]["std"])
    # dt exists but in dT mode we don't use it in loss directly
    return {"T_mean": Tm, "T_std": Ts, "dT_mean": dTm, "dT_std": dTs}

def _teacher_prob(epoch_idx_1based: int, cfg: TrainConfig) -> float:
    # linear decay from start -> end over teacher_decay_epochs
    e = max(0, epoch_idx_1based - 1)
    if cfg.teacher_decay_epochs <= 0:
        return float(cfg.teacher_end)
    t = min(1.0, e / float(cfg.teacher_decay_epochs))
    return float(cfg.teacher_start * (1.0 - t) + cfg.teacher_end * t)

def _early_weight(t_frac: float, cfg: TrainConfig) -> float:
    if t_frac <= cfg.early_t0:
        return float(cfg.early_w0)
    if t_frac <= cfg.early_t1:
        return float(cfg.early_w1)
    return 1.0


# ---------- SDF helper ----------
def _sdf_to_boundary(pos: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
    """
    Signed distance approximation: distance to nearest boundary node (>=0).
    (Знака нет, используем просто distance; для MVP достаточно.)
    """
    # pos: [N,3], boundary_mask: [N] bool
    b = boundary_mask.nonzero(as_tuple=False).view(-1)
    if b.numel() == 0:
        return torch.full((pos.size(0),), 1.0, dtype=pos.dtype)
    pb = pos[b]  # [Nb,3]

    # Fast path: scipy KDTree if available
    try:
        import numpy as np
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(pb.detach().cpu().numpy())
        d, _ = tree.query(pos.detach().cpu().numpy(), k=1)
        return torch.from_numpy(d).to(pos.device, dtype=pos.dtype)
    except Exception:
        # fallback: torch cdist with chunking
        N = pos.size(0)
        out = torch.empty((N,), device=pos.device, dtype=pos.dtype)
        chunk = 1024
        for i in range(0, N, chunk):
            pi = pos[i : i + chunk]  # [c,3]
            d = torch.cdist(pi, pb)  # [c,Nb]
            out[i : i + chunk] = d.min(dim=1).values
        return out


# ---------- Dataset (self-contained, reads your .pt schema) ----------
class PTUnrollDataset(Dataset):
    """
    Reads dict .pt with keys: pos[N,3], edge_index[2,E], markers_oh[N,3], temps[T,N], times[T]
    Returns PyG Data with:
      x[N,8] = pos(3), markers(3), T_current(1), sdf(1)
      edge_attr[E,4] = dpos(3), dist(1)
      T_seq[N,K+1] physical
      dT_seq[N,K] physical
      t_frac scalar (float)
      boundary_mask[N] bool
    """

    def __init__(
        self,
        files: List[str],
        unroll_k: int,
        time_mode: str,
        fixed_t: int,
        fixed_frac: Optional[float],
        seed: int,
    ):
        self.files = files
        self.K = int(unroll_k)
        self.time_mode = time_mode
        self.fixed_t = int(fixed_t)
        self.fixed_frac = fixed_frac
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def _choose_t0(self, T_total: int) -> Tuple[int, float]:
        # Need t0 so that t0+K <= T_total-1
        max_t0 = max(0, (T_total - 1) - self.K)
        if max_t0 <= 0:
            return 0, 0.0

        if self.time_mode == "fixed":
            t0 = min(self.fixed_t, max_t0)
            return t0, t0 / float(T_total - 1)

        if self.time_mode == "frac":
            frac = float(self.fixed_frac if self.fixed_frac is not None else 0.0)
            t0 = int(math.floor(frac * max_t0))
            t0 = max(0, min(max_t0, t0))
            return t0, t0 / float(T_total - 1)

        if self.time_mode == "mix_early":
            # bias to early region 70% in first 40% timeline
            if self.rng.random() < 0.7:
                t0 = self.rng.randint(0, max(0, int(0.4 * max_t0)))
            else:
                t0 = self.rng.randint(0, max_t0)
            return t0, t0 / float(T_total - 1)

        # default: random
        t0 = self.rng.randint(0, max_t0)
        return t0, t0 / float(T_total - 1)

    def __getitem__(self, idx: int) -> Data:
        path = self.files[idx]
        d = torch.load(path, map_location="cpu")  # dict
        pos = d["pos"].float()  # [N,3]
        edge_index = d["edge_index"].long()  # [2,E]
        markers = d["markers_oh"].float()  # [N,3]
        temps = d["temps"].float()  # [T,N] in your log
        times = d.get("times", None)
        if times is not None:
            times = times.float()

        T_total = temps.size(0)
        t0, t_frac = self._choose_t0(T_total)

        # Build sequences
        # temps: [T,N] -> take [K+1, N] -> transpose to [N,K+1]
        T_seq = temps[t0 : t0 + self.K + 1].transpose(0, 1).contiguous()  # [N,K+1]
        dT_seq = T_seq[:, 1:] - T_seq[:, :-1]  # [N,K]

        # Boundary + SDF
        boundary_mask = (markers.sum(dim=1) > 0.0)
        sdf = _sdf_to_boundary(pos, boundary_mask).unsqueeze(1)  # [N,1]

        # Node x (T_current will be overwritten per-step in train loop; init with T_seq[:,0])
        T0 = T_seq[:, 0:1]
        x = torch.cat([pos, markers, T0, sdf], dim=1)  # [N,8]

        # Edge attr
        src = edge_index[0]
        dst = edge_index[1]
        dpos = pos[dst] - pos[src]  # [E,3]
        dist = torch.norm(dpos, dim=1, keepdim=True)  # [E,1]
        edge_attr = torch.cat([dpos, dist], dim=1)  # [E,4]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.T_seq = T_seq  # [N,K+1]
        data.dT_seq = dT_seq  # [N,K]
        data.t_frac = torch.tensor([t_frac], dtype=torch.float32)
        data.boundary_mask = boundary_mask

        # optional: keep filename for debugging
        data.fname = os.path.basename(path)
        return data


# ---------- Model picker / builder ----------
def _pick_model_class(model_mod):
    candidates = []
    for name, obj in vars(model_mod).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if "MeshGraphNet" in name and "Config" not in name:
                candidates.append((name, obj))
    if not candidates:
        # fallback: any nn.Module subclass
        for name, obj in vars(model_mod).items():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                candidates.append((name, obj))
    if not candidates:
        raise ImportError("Не нашёл ни одного класса модели (nn.Module) в src.model")

    # Prefer Delta / dT model if exists
    pref = ["MeshGraphNetDelta", "MeshGraphNetDT", "MeshGraphNet"]
    for p in pref:
        for name, cls in candidates:
            if name == p:
                return name, cls
    return candidates[0]

def _build_cfg_object(cfg_cls, train_cfg: TrainConfig):
    # canonical
    canon = {
        "in_dim": 8,
        "edge_dim": 4,
        "hidden_dim": train_cfg.hidden_dim,
        "num_blocks": train_cfg.num_blocks,
        "mlp_layers": train_cfg.mlp_layers,
        "use_checkpoint": train_cfg.use_checkpoint,
        "out_dim": 1,
        "dropout": 0.0,
    }
    synonyms = {
        "in_node": canon["in_dim"],
        "in_edge": canon["edge_dim"],
        "hidden": canon["hidden_dim"],
        "mp_steps": canon["num_blocks"],
        "checkpoint": canon["use_checkpoint"],
        "grad_checkpoint": canon["use_checkpoint"],
    }

    sig = inspect.signature(cfg_cls.__init__)
    pset = set(sig.parameters.keys())
    pset.discard("self")

    kwargs = {}
    for k, v in canon.items():
        if k in pset:
            kwargs[k] = v
    for k, v in synonyms.items():
        if k in pset and k not in kwargs:
            kwargs[k] = v

    return cfg_cls(**kwargs)

def build_model(cfg: TrainConfig) -> nn.Module:
    model_mod = importlib.import_module("src.model")
    name, cls = _pick_model_class(model_mod)
    print(f"[model] picked class from src.model: {name}")

    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]

    # Pattern: __init__(self, cfg: MeshGraphNetConfig)
    if len(params) == 1 and params[0].name == "cfg":
        cfg_type = None
        ann = params[0].annotation
        if isinstance(ann, str):
            cfg_type = getattr(model_mod, ann, None)
        else:
            cfg_type = ann if ann is not inspect._empty else None

        if cfg_type is None:
            cfg_type = getattr(model_mod, "MeshGraphNetConfig", None)

        if cfg_type is None:
            raise TypeError(
                f"Model '{name}' требует cfg, но MeshGraphNetConfig не найден в src.model"
            )

        cfg_obj = _build_cfg_object(cfg_type, cfg)
        print(f"[model] building {name} with cfg={cfg_type.__name__}({cfg_obj})")
        return cls(cfg_obj)

    # Pattern: kwargs
    kwargs = {}
    pset = set(sig.parameters.keys())
    pset.discard("self")

    # common kwargs
    for k, v in {
        "in_dim": 8,
        "edge_dim": 4,
        "hidden_dim": cfg.hidden_dim,
        "num_blocks": cfg.num_blocks,
        "mlp_layers": cfg.mlp_layers,
        "use_checkpoint": cfg.use_checkpoint,
        "out_dim": 1,
    }.items():
        if k in pset:
            kwargs[k] = v

    # synonyms
    for k, v in {
        "in_node": 8,
        "in_edge": 4,
        "hidden": cfg.hidden_dim,
        "mp_steps": cfg.num_blocks,
        "checkpoint": cfg.use_checkpoint,
    }.items():
        if k in pset and k not in kwargs:
            kwargs[k] = v

    try:
        return cls(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Не удалось создать модель {name} kwargs={kwargs}\n"
            f"sig={sig}\n"
            f"err={e}"
        )


# ---------- Loss helpers ----------
def huber_elementwise(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    # returns elementwise huber
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="none")

def boundary_weights(boundary_mask: torch.Tensor, boundary_w: float) -> torch.Tensor:
    # w = 1 + boundary_w * is_boundary
    return 1.0 + float(boundary_w) * boundary_mask.float()


# ---------- Validation ----------
@torch.no_grad()
def evaluate_unroll(
    model: nn.Module,
    loader,
    device: torch.device,
    stats: Dict[str, float],
    cfg: TrainConfig,
    use_ema: Optional[EMA] = None,
) -> Dict[str, float]:
    model.eval()
    if use_ema is not None:
        use_ema.apply_to(model)

    T_mean = stats["T_mean"]
    T_std = max(stats["T_std"], 1e-8)
    dT_mean = stats["dT_mean"]
    dT_std = max(stats["dT_std"], 1e-8)

    T_COL = 6  # pos(3) markers(3) -> index 6 is T_current
    K = cfg.unroll_k

    total_TK_se = 0.0
    total_TK_ae = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x0 = batch.x.clone()
        T_seq = batch.T_seq  # [N,K+1] physical
        boundary_mask = getattr(batch, "boundary_mask", None)
        if boundary_mask is None:
            boundary_mask = (x0[:, 3:6].sum(dim=1) > 0)

        # start from GT T0
        T_pred = T_seq[:, 0].clone()  # [N]
        for j in range(K):
            # update normalized T in x
            Tn = (T_pred - T_mean) / T_std
            Tn = torch.clamp(Tn, -cfg.clip_t_norm, cfg.clip_t_norm)
            x = x0.clone()
            x[:, T_COL] = Tn

            with _autocast(enabled=False, device=device):
                out = model(x, batch.edge_index, batch.edge_attr)

            if out.dim() == 2 and out.size(-1) == 1:
                out = out.view(-1)

            dTn_pred = out
            dTn_pred = torch.clamp(dTn_pred, -cfg.clip_dt_norm, cfg.clip_dt_norm)
            dT_pred = dTn_pred * dT_std + dT_mean
            T_pred = T_pred + dT_pred

        # TK error vs GT
        T_gt = T_seq[:, K]
        se = (T_pred - T_gt) ** 2
        ae = (T_pred - T_gt).abs()
        total_TK_se += se.sum().item()
        total_TK_ae += ae.sum().item()
        total_nodes += T_gt.numel()

    mse = total_TK_se / max(1, total_nodes)
    mae = total_TK_ae / max(1, total_nodes)

    if use_ema is not None:
        use_ema.restore(model)

    return {"TK_mse": mse, "TK_mae": mae}


# ---------- Training ----------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    ensure_dir(cfg.weights_dir)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Stats for global T norm + dT norm
    stats = _load_stats(cfg.delta_stats_path)
    T_mean = stats["T_mean"]
    T_std = max(stats["T_std"], 1e-8)
    dT_mean = stats["dT_mean"]
    dT_std = max(stats["dT_std"], 1e-8)

    # files / split
    files = _list_pt_files(cfg.data_dir)
    train_files, val_files = _split_files(files, cfg.val_ratio, cfg.seed)

    # datasets
    train_ds = PTUnrollDataset(
        files=train_files,
        unroll_k=cfg.unroll_k,
        time_mode=cfg.time_mode,
        fixed_t=cfg.fixed_t,
        fixed_frac=None,
        seed=cfg.seed,
    )

    # val loaders per frac
    val_loaders = {}
    for frac in cfg.val_fracs:
        ds = PTUnrollDataset(
            files=val_files,
            unroll_k=cfg.unroll_k,
            time_mode="frac",
            fixed_t=0,
            fixed_frac=float(frac),
            seed=cfg.seed,
        )
        val_loaders[float(frac)] = ds

    # loaders
    # batch_size must stay 1 (your VRAM, unroll_k=4)
    train_loader = PyGDataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    val_pyg_loaders = {
        frac: PyGDataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, cfg.num_workers // 2),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(cfg.num_workers > 0),
        )
        for frac, ds in val_loaders.items()
    }

    # model
    model = build_model(cfg).to(device)

    # optimizer / sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr_min
    )

    scaler = _GradScaler(enabled=(cfg.use_amp and device.type == "cuda"), device=device)
    use_amp = (cfg.use_amp and device.type == "cuda")

    ema = EMA(model, decay=cfg.ema_decay)

    # checkpoint paths
    resume_path = os.path.join(cfg.weights_dir, f"{cfg.run_name}_resume.pt")
    best_path = os.path.join(cfg.weights_dir, f"{cfg.run_name}_best.pt")

    start_epoch = 1
    best_metric = float("inf")

    # --- init_from or resume ---
    def _load_model_only(path: str):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            # could be state_dict directly
            sd = ckpt
        else:
            raise ValueError(f"Unknown checkpoint format: {path}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded model from: {path}")
        if missing:
            print(f"[init] missing keys: {len(missing)}")
        if unexpected:
            print(f"[init] unexpected keys: {len(unexpected)}")

    if cfg.auto_resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        ema.shadow = ckpt.get("ema", ema.shadow)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        if not cfg.reset_optim_on_resume:
            optimizer.load_state_dict(ckpt["optim"])
            scheduler.load_state_dict(ckpt["sched"])
            if "scaler" in ckpt and use_amp:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
        print(f"[resume] loaded checkpoint: {resume_path}")
        print(f"[resume] start_epoch={start_epoch}, best_metric={best_metric}")
        print(f"[resume] reset_optim={cfg.reset_optim_on_resume}")
    elif cfg.init_from:
        _load_model_only(cfg.init_from)
        print(f"[init] fine-tune from init_from, reset_optim={cfg.reset_optim_on_resume}")

    # baseline val (dT=0 => T stays constant; evaluate TK error)
    # For baseline: just run evaluate_unroll with model that outputs zero.
    # We'll print baseline using "constant T" for quick sanity:
    print("[baseline val] dT=0 => evaluate TK error")
    for frac, vloader in val_pyg_loaders.items():
        # compute TK error if T_pred = T0
        total_se = 0.0
        total_ae = 0.0
        total_n = 0
        for batch in vloader:
            T_seq = batch.T_seq  # [N,K+1] physical
            T0 = T_seq[:, 0]
            TK = T_seq[:, cfg.unroll_k]
            se = (T0 - TK) ** 2
            ae = (T0 - TK).abs()
            total_se += se.sum().item()
            total_ae += ae.sum().item()
            total_n += TK.numel()
        mse0 = total_se / max(1, total_n)
        mae0 = total_ae / max(1, total_n)
        print(f"  val_frac_{frac:.2f}: TK_mse0={mse0:.6e} TK_mae0={mae0:.6e} fallback=0.00%")

    # training loop
    T_COL = 6
    K = cfg.unroll_k
    bad_batches = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0_epoch = time.time()
        p_teacher = _teacher_prob(epoch, cfg)

        running_loss = 0.0
        running_nodes = 0
        bad_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            x_base = batch.x  # [N,8], contains T0 raw currently (physical)
            T_seq = batch.T_seq  # [N,K+1] physical
            dT_seq = batch.dT_seq  # [N,K] physical
            t_frac = float(batch.t_frac.item()) if hasattr(batch, "t_frac") else 0.0
            early_w = _early_weight(t_frac, cfg)

            # boundary weighting
            bmask = getattr(batch, "boundary_mask", None)
            if bmask is None:
                bmask = (x_base[:, 3:6].sum(dim=1) > 0)
            w_node = boundary_weights(bmask, cfg.boundary_w)  # [N]

            # Unroll
            # Start with GT T0
            T_curr = T_seq[:, 0].clone()  # [N] physical
            x0 = x_base.clone()

            total_loss = 0.0

            for j in range(K):
                # Normalize T for input
                Tn = (T_curr - T_mean) / T_std
                Tn = torch.clamp(Tn, -cfg.clip_t_norm, cfg.clip_t_norm)

                x = x0.clone()
                x[:, T_COL] = Tn

                # model forward -> predicts normalized dT
                with _autocast(enabled=use_amp, device=device):
                    out = model(x, batch.edge_index, batch.edge_attr)

                if out.dim() == 2 and out.size(-1) == 1:
                    out = out.view(-1)
                dTn_pred = torch.clamp(out, -cfg.clip_dt_norm, cfg.clip_dt_norm)

                # target normalized dT
                dT_gt = dT_seq[:, j]
                dTn_gt = (dT_gt - dT_mean) / dT_std
                dTn_gt = torch.clamp(dTn_gt, -cfg.clip_dt_norm, cfg.clip_dt_norm)

                # loss per node (huber) with boundary weights
                l_elem = huber_elementwise(dTn_pred, dTn_gt, beta=cfg.huber_beta)  # [N]
                l_elem = l_elem * w_node

                # gamma weighting across steps
                step_w = (cfg.unroll_gamma ** j)
                total_loss = total_loss + step_w * l_elem.mean()

                # advance
                # predicted physical dT
                dT_pred = dTn_pred * dT_std + dT_mean
                T_next_pred = T_curr + dT_pred

                # teacher forcing
                use_gt = (random.random() < p_teacher)
                if use_gt:
                    T_curr = T_seq[:, j + 1].clone()
                else:
                    T_curr = T_next_pred

            total_loss = total_loss * early_w

            # NaN guard
            if not torch.isfinite(total_loss):
                bad_batches += 1
                if cfg.disable_amp_on_nan and use_amp:
                    # turn off AMP for stability
                    use_amp = False
                    try:
                        scaler = _GradScaler(enabled=False, device=device)
                    except Exception:
                        pass
                    print(f"[nan] detected -> disable AMP (epoch={epoch} it={it})")
                if bad_batches >= cfg.max_bad_batches_per_epoch:
                    raise RuntimeError(
                        f"Too many bad batches (NaN/Inf) in epoch {epoch}: {bad_batches}"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            # backward (accum)
            loss_scaled = total_loss / max(1, cfg.accum_steps)

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # step
            if (it % cfg.accum_steps) == 0:
                if cfg.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                ema.update(model)

            running_loss += float(total_loss.detach().item()) * batch.num_nodes
            running_nodes += int(batch.num_nodes)

        # end epoch
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        train_loss = running_loss / max(1, running_nodes)

        # validation across fracs (use EMA weights)
        val_metrics = {}
        val_mse_mean = 0.0
        for frac, vloader in val_pyg_loaders.items():
            m = evaluate_unroll(model, vloader, device, stats, cfg, use_ema=ema)
            val_metrics[frac] = m
            val_mse_mean += m["TK_mse"]
        val_mse_mean /= max(1, len(val_pyg_loaders))

        # choose best metric = mean TK mse (stable)
        metric = val_mse_mean

        # print
        peak_alloc = 0.0
        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()

        dt_epoch = time.time() - t0_epoch
        print(
            f"[epoch {epoch:04d}] train_loss={train_loss:.6e} "
            f"lr={lr_now:.2e} ema={cfg.ema_decay:.4f} p_teacher={p_teacher:.3f} "
            f"val_TK_mse_mean={val_mse_mean:.6e} peak_alloc={peak_alloc:.2f} GB "
            f"time={dt_epoch:.1f}s"
        )
        for frac in cfg.val_fracs:
            m = val_metrics[float(frac)]
            print(
                f"  val_frac_{float(frac):.2f}: TK_mse={m['TK_mse']:.6e} TK_mae={m['TK_mae']:.6e} fallback=0.00%"
            )

        # save best
        if metric < best_metric:
            best_metric = metric
            # save EMA weights as best
            ema.apply_to(model)
            torch.save({"model": model.state_dict(), "best_metric": best_metric, "epoch": epoch, "cfg": asdict(cfg)}, best_path)
            ema.restore(model)
            print(f"[save] best -> {best_path} (best_metric={best_metric:.6f})")

        # save resume
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.shadow,
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "cfg": asdict(cfg),
            },
            resume_path,
        )

    print(f"[done] best checkpoint: {best_path} (best_metric={best_metric})")
