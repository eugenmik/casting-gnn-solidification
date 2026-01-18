# src/dataset.py
from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


IDX_T = 6  # in x: pos(0:3), markers(3:6), T(6), sdf(7)
IDX_SDF = 7


def _safe_std(x: float, eps: float = 1e-12) -> float:
    return float(x) if float(x) > eps else 1.0


def load_delta_stats(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _compute_sdf(pos: torch.Tensor, is_boundary: torch.Tensor) -> torch.Tensor:
    """
    Signed distance to nearest boundary node (always >=0 here; 'signed' placeholder).
    Uses cKDTree if available; falls back to torch.cdist.
    """
    pos = pos.contiguous()
    bidx = torch.where(is_boundary)[0]
    if bidx.numel() == 0:
        return torch.zeros((pos.size(0),), dtype=pos.dtype)

    bpos = pos[bidx].cpu().numpy()
    ppos = pos.cpu().numpy()

    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(bpos)
        d, _ = tree.query(ppos, k=1, workers=-1)
        return torch.from_numpy(d).to(dtype=pos.dtype)
    except Exception:
        # fallback
        d = torch.cdist(pos[bidx], pos)  # [Nb, N]
        sdf = d.min(dim=0).values
        return sdf


@dataclass
class Split:
    train_files: List[str]
    val_files: List[str]


def split_files(files: List[str], val_ratio: float, seed: int) -> Split:
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_val = max(1, int(round(n * val_ratio)))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return Split(train_files=train_files, val_files=val_files)


class CastingUnrollDataset(Dataset):
    """
    Produces PyG Data with:
      - x: [N,8] (pos, markers_oh, T_norm_global, sdf)
      - edge_index: [2,E]
      - edge_attr: [E,4] (dx,dy,dz,dist)
      - T_seq: [N, K+1] physical units
      - dT_seq: [N, K] normalized (global dT stats)
      - boundary: [N] 0/1
      - t_frac: [N] scalar repeated (for early weighting)
    """

    def __init__(
        self,
        files: List[str],
        delta_stats_path: str,
        time_mode: str = "random",
        fixed_t: int = 0,
        frac: float = 0.0,
        unroll_k: int = 4,
        mix_early_frac: float = 0.25,
        t_norm_eps: float = 1e-3,
        clip_t_norm: float = 10.0,
    ):
        self.files = files
        self.time_mode = time_mode
        self.fixed_t = int(fixed_t)
        self.frac = float(frac)
        self.unroll_k = int(unroll_k)
        self.mix_early_frac = float(mix_early_frac)

        stats = load_delta_stats(delta_stats_path)
        self.T_mean = float(stats["T_current"]["mean"])
        self.T_std = _safe_std(stats["T_current"]["std"])
        self.dT_mean = float(stats["dT"]["mean"])
        self.dT_std = _safe_std(stats["dT"]["std"])

        self.t_norm_eps = float(t_norm_eps)
        self.clip_t_norm = float(clip_t_norm)

        # simple SDF cache (per file)
        self._sdf_cache: Dict[str, torch.Tensor] = {}
        self._sdf_cache_order: List[str] = []
        self._sdf_cache_max = 128

    def __len__(self) -> int:
        return len(self.files)

    def _cache_put(self, key: str, value: torch.Tensor) -> None:
        if key in self._sdf_cache:
            return
        self._sdf_cache[key] = value
        self._sdf_cache_order.append(key)
        if len(self._sdf_cache_order) > self._sdf_cache_max:
            k = self._sdf_cache_order.pop(0)
            self._sdf_cache.pop(k, None)

    def _sample_t0(self, T: int) -> int:
        K = self.unroll_k
        max_t0 = T - (K + 1)
        if max_t0 < 0:
            raise ValueError(f"Sequence too short: T={T}, need >= K+1 where K={K}")

        if self.time_mode == "fixed":
            return max(0, min(self.fixed_t, max_t0))

        if self.time_mode == "frac":
            return int(math.floor(self.frac * max(1, max_t0)))

        if self.time_mode == "mix_early":
            # with prob 0.6 sample from early region
            if random.random() < 0.6:
                hi = max(0, int(math.floor(self.mix_early_frac * max_t0)))
                return random.randint(0, hi) if hi > 0 else 0
            return random.randint(0, max_t0)

        # default random
        return random.randint(0, max_t0)

    def __getitem__(self, idx: int) -> Data:
        path = self.files[idx]
        d = torch.load(path, map_location="cpu")

        pos = d["pos"].to(torch.float32)                 # [N,3]
        edge_index = d["edge_index"].to(torch.int64)     # [2,E]
        markers = d["markers_oh"].to(torch.float32)      # [N,3]
        temps = d["temps"].to(torch.float32)             # [T,N]
        times = d.get("times", None)

        T_total = temps.size(0)
        t0 = self._sample_t0(T_total)
        K = self.unroll_k

        # sequences in physical units
        T_seq = temps[t0 : t0 + K + 1].transpose(0, 1).contiguous()  # [N,K+1]
        dT_phys = (temps[t0 + 1 : t0 + K + 1] - temps[t0 : t0 + K]).transpose(0, 1).contiguous()  # [N,K]

        # normalize dT target
        dT_norm = (dT_phys - self.dT_mean) / (self.dT_std + 1e-12)
        dT_norm = dT_norm.clamp(-10.0, 10.0)

        # boundary
        is_boundary = (markers.sum(dim=1) > 0.0)

        # sdf (cached)
        if path in self._sdf_cache:
            sdf = self._sdf_cache[path]
        else:
            sdf = _compute_sdf(pos, is_boundary)
            self._cache_put(path, sdf)
        sdf = sdf.to(torch.float32)

        # global T norm for step0
        T0 = T_seq[:, 0]
        T0_norm = (T0 - self.T_mean) / (self.T_std + 1e-12)
        T0_norm = T0_norm.clamp(-self.clip_t_norm, self.clip_t_norm)

        # node features x: 8 dims
        x = torch.cat([pos, markers, T0_norm[:, None], sdf[:, None]], dim=1).contiguous()

        # edge_attr: rel displacement + dist
        src, dst = edge_index[0], edge_index[1]
        rel = (pos[dst] - pos[src]).to(torch.float32)              # [E,3]
        dist = torch.norm(rel, dim=1, keepdim=True)                # [E,1]
        edge_attr = torch.cat([rel, dist], dim=1).contiguous()     # [E,4]

        # t_frac per node (for early weighting)
        max_t0 = max(1, T_total - (K + 1))
        t_frac = float(t0) / float(max_t0)
        t_frac_node = torch.full((pos.size(0),), t_frac, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.pos = pos
        data.markers_oh = markers
        data.boundary = is_boundary.to(torch.float32)          # [N]
        data.T_seq = T_seq                                    # [N,K+1] physical
        data.dT_seq = dT_norm                                 # [N,K] normalized
        data.t_frac = t_frac_node                             # [N]
        if times is not None:
            data.times = times.to(torch.float32)
        return data


def list_pt_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files in: {data_dir}")
    return files
