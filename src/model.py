# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

try:
    from torch_scatter import scatter_add  # type: ignore
except Exception:
    from torch_geometric.utils import scatter  # type: ignore
    def scatter_add(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class EdgeMLPNoConcat(nn.Module):
    """
    Implements an MLP equivalent to:
      MLP([h_src, h_dst, e])  without explicitly concatenating [E, 3H].
    This saves a large temporary allocation and prevents OOM.
    """
    def __init__(self, hidden_dim: int, mlp_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_layers = mlp_layers

        # First linear of a concat-MLP can be decomposed as sum of three linears:
        # W*[h_src, h_dst, e] = W1*h_src + W2*h_dst + W3*e
        self.lin_src = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.lin_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_e   = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Additional hidden layers (if mlp_layers > 2)
        hidden = []
        for _ in range(max(0, mlp_layers - 2)):
            hidden += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.hidden = nn.Sequential(*hidden)

        # Output layer (to match a 2-layer MLP structure)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, h_src: Tensor, h_dst: Tensor, e: Tensor) -> Tensor:
        z = self.lin_src(h_src) + self.lin_dst(h_dst) + self.lin_e(e)  # [E, H]
        z = self.act(z)
        z = self.hidden(z)
        return self.out(z)


class GraphNetBlock(nn.Module):
    """
    MeshGraphNet processor block with residual updates.
    Memory-safe edge update avoids explicit [E, 3H] concatenation.
    """
    def __init__(self, hidden_dim: int, mlp_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_mlp = EdgeMLPNoConcat(hidden_dim, mlp_layers=mlp_layers)
        self.node_mlp = make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, num_layers=mlp_layers)

        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, e: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]

        de = self.edge_mlp(h[src], h[dst], e)  # [E, H]
        e = self.edge_norm(e + de)

        agg = scatter_add(e, dst, dim=0, dim_size=h.size(0))  # [N, H]
        dh = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.node_norm(h + dh)

        return h, e


@dataclass
class MeshGraphNetConfig:
    node_in_dim: int = 8
    edge_in_dim: int = 4
    hidden_dim: int = 128
    num_processor_blocks: int = 15
    mlp_layers: int = 2
    out_dim: int = 1
    use_checkpoint: bool = True  # critical for VRAM


class MeshGraphNetDelta(nn.Module):
    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        self.cfg = cfg

        self.node_encoder = make_mlp(cfg.node_in_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=cfg.mlp_layers)
        self.edge_encoder = make_mlp(cfg.edge_in_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=cfg.mlp_layers)

        self.processors = nn.ModuleList(
            [GraphNetBlock(cfg.hidden_dim, mlp_layers=cfg.mlp_layers) for _ in range(cfg.num_processor_blocks)]
        )

        self.decoder = make_mlp(cfg.hidden_dim, cfg.out_dim, cfg.hidden_dim, num_layers=cfg.mlp_layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.node_encoder(x)          # [N, H]
        e = self.edge_encoder(edge_attr)  # [E, H]

        if self.cfg.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            for block in self.processors:
                def _fn(h_in: Tensor, e_in: Tensor) -> Tuple[Tensor, Tensor]:
                    return block(h_in, e_in, edge_index)
                h, e = checkpoint(_fn, h, e, use_reentrant=False)
        else:
            for block in self.processors:
                h, e = block(h, e, edge_index)

        out = self.decoder(h)             # [N, 1]
        return out.squeeze(-1)            # [N]
