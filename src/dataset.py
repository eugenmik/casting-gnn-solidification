"""
src/dataset.py

Loads graph+time-series samples saved as torch .pt files.

Expected file format (torch.load):
{
  'pos':        FloatTensor [N, 3]
  'edge_index': LongTensor  [2, E]
  'markers_oh': FloatTensor [N, M]  (M=3 in your dataset)
  'temps':      FloatTensor [S, N]
  'times':      FloatTensor [S]
}

This dataset returns a dict with:
- pos, pos_norm
- edge_index, edge_attr (computed from pos_norm)
- markers_oh, boundary_mask
- temps, times
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def list_pt_files(data_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(data_dir):
        if fn.endswith(".pt"):
            files.append(os.path.join(data_dir, fn))
    files.sort()
    return files


def normalize_pos(pos: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-graph normalization:
    - center by centroid
    - scale by max range (bbox diagonal-ish)
    """
    center = pos.mean(dim=0, keepdim=True)
    p = pos - center
    # scale by max norm to keep values in a stable range
    scale = torch.linalg.norm(p, dim=1).max().clamp_min(eps)
    return p / scale


def build_edge_attr(pos_norm: torch.Tensor, edge_index: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Edge features: [dpos(3), dist(1)] from normalized positions.
    """
    src = edge_index[0]
    dst = edge_index[1]
    dpos = pos_norm[dst] - pos_norm[src]  # [E, 3]
    dist = torch.linalg.norm(dpos, dim=1, keepdim=True).clamp_min(eps)  # [E, 1]
    edge_attr = torch.cat([dpos, dist], dim=1).to(torch.float32)        # [E, 4]
    return edge_attr


@dataclass
class DatasetMeta:
    marker_dim: int
    steps: int


class GraphSequenceDataset(Dataset):
    def __init__(self, data_dir: str, files: List[str] | None = None):
        super().__init__()
        self.data_dir = data_dir
        self.files = files if files is not None else list_pt_files(data_dir)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in: {data_dir}")

        # quick meta from first file
        sample = torch.load(self.files[0], map_location="cpu")
        self.meta = DatasetMeta(marker_dim=int(sample["markers_oh"].shape[1]), steps=int(sample["temps"].shape[0]))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        d = torch.load(path, map_location="cpu")

        pos = d["pos"].to(torch.float32)
        edge_index = d["edge_index"].to(torch.int64)
        markers_oh = d["markers_oh"].to(torch.float32)
        temps = d["temps"].to(torch.float32)  # [S, N]
        times = d["times"].to(torch.float32)  # [S]

        pos_norm = normalize_pos(pos)
        edge_attr = build_edge_attr(pos_norm, edge_index)

        # boundary heuristic: any marker active => boundary
        boundary_mask = (markers_oh.sum(dim=1) > 0.5).to(torch.bool)

        return {
            "path": path,
            "pos": pos,
            "pos_norm": pos_norm,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "markers_oh": markers_oh,
            "boundary_mask": boundary_mask,
            "temps": temps,
            "times": times,
        }
