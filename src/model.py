"""
src/model.py

MeshGraphNet-style Graph Network for predicting per-node dT (temperature delta)
on a mesh/graph.

Key points:
- Node features: [pos_norm(3), markers_oh(M), T_norm(1), dt_norm(1)] -> default M=3 => 8 dims
- Edge features: [dpos_norm(3), dist_norm(1)] -> 4 dims
- Output: per-node dT_norm (1)

Forward is flexible:
- model(x, edge_index, edge_attr)
- model(data_dict) where dict has keys x/edge_index/edge_attr or pos/markers_oh/etc (best effort)
- model(torch_geometric.data.Data) with attributes x, edge_index, edge_attr
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_add
except Exception as e:
    raise ImportError("torch_scatter is required for this model. Install PyG deps.") from e


# -----------------------------
# Helpers
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layers: int = 2, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        assert layers >= 1
        mods = []
        d0 = in_dim
        for i in range(layers - 1):
            mods.append(nn.Linear(d0, hidden_dim))
            mods.append(nn.SiLU())
            d0 = hidden_dim
        mods.append(nn.Linear(d0, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MeshGraphNetConfig:
    node_in_dim: int = 8
    edge_in_dim: int = 4
    hidden_dim: int = 128
    num_processor_blocks: int = 15
    mlp_layers: int = 2
    out_dim: int = 1
    use_checkpoint: bool = True


class GNBlock(nn.Module):
    """
    Graph Network block (edge update + node update) with residual connections.
    """

    def __init__(self, hidden_dim: int, mlp_layers: int = 2):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, layers=mlp_layers, hidden_dim=hidden_dim)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, layers=mlp_layers, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [N, H]
        edge_index: [2, E] with src=edge_index[0], dst=edge_index[1]
        e: [E, H]
        """
        src = edge_index[0]
        dst = edge_index[1]

        e_in = torch.cat([e, x[src], x[dst]], dim=-1)  # [E, 3H]
        e_up = self.edge_mlp(e_in)                     # [E, H]

        # aggregate messages to destination nodes
        agg = scatter_add(e_up, dst, dim=0, dim_size=x.size(0))  # [N, H]

        x_in = torch.cat([x, agg], dim=-1)  # [N, 2H]
        x_up = self.node_mlp(x_in)          # [N, H]

        # residual updates
        return x + x_up, e + e_up


class MeshGraphNetDelta(nn.Module):
    """
    Predicts per-node dT_norm.
    """

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        self.cfg = cfg

        self.node_enc = MLP(cfg.node_in_dim, cfg.hidden_dim, layers=cfg.mlp_layers, hidden_dim=cfg.hidden_dim)
        self.edge_enc = MLP(cfg.edge_in_dim, cfg.hidden_dim, layers=cfg.mlp_layers, hidden_dim=cfg.hidden_dim)

        self.blocks = nn.ModuleList(
            [GNBlock(cfg.hidden_dim, mlp_layers=cfg.mlp_layers) for _ in range(cfg.num_processor_blocks)]
        )
        self.node_dec = MLP(cfg.hidden_dim, cfg.out_dim, layers=cfg.mlp_layers, hidden_dim=cfg.hidden_dim)

    def _unpack_inputs(
        self,
        x_or_data: Union[torch.Tensor, Dict[str, Any], Any],
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accept:
        - (x, edge_index, edge_attr)
        - dict with keys x/edge_index/edge_attr
        - PyG Data with attributes x/edge_index/edge_attr
        """
        if isinstance(x_or_data, torch.Tensor):
            assert edge_index is not None and edge_attr is not None
            return x_or_data, edge_index, edge_attr

        # dict-like
        if isinstance(x_or_data, dict):
            if "x" in x_or_data and "edge_index" in x_or_data and "edge_attr" in x_or_data:
                return x_or_data["x"], x_or_data["edge_index"], x_or_data["edge_attr"]
            raise TypeError(
                "MeshGraphNetDelta.forward(dict) requires keys: x, edge_index, edge_attr. "
                "Build features in training/visualization before calling the model."
            )

        # PyG Data-like
        if hasattr(x_or_data, "x") and hasattr(x_or_data, "edge_index") and hasattr(x_or_data, "edge_attr"):
            return x_or_data.x, x_or_data.edge_index, x_or_data.edge_attr

        raise TypeError(
            "Unsupported input type for MeshGraphNetDelta.forward. "
            "Use forward(x, edge_index, edge_attr) or pass a Data/dict with x/edge_index/edge_attr."
        )

    def forward(
        self,
        x_or_data: Union[torch.Tensor, Dict[str, Any], Any],
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, edge_index, edge_attr = self._unpack_inputs(x_or_data, edge_index, edge_attr)

        # encode
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)

        # process
        if self.cfg.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            for blk in self.blocks:
                h, e = checkpoint(lambda hh, ee: blk(hh, edge_index, ee), h, e)
        else:
            for blk in self.blocks:
                h, e = blk(h, edge_index, e)

        # decode
        out = self.node_dec(h)  # [N, out_dim]
        if out.shape[-1] == 1:
            out = out.squeeze(-1)  # [N]
        return out
