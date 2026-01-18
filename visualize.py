# visualize.py
from __future__ import annotations

import argparse
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
import pyvista as pv

from src.model import MeshGraphNetDelta, MeshGraphNetConfig


def robust_T_norm(T: torch.Tensor, eps: float = 1e-3, clip: float = 10.0) -> torch.Tensor:
    mu = T.mean()
    sd = T.std(unbiased=False).clamp_min(eps)
    Tn = (T - mu) / sd
    return Tn.clamp(-clip, clip)


def load_delta_stats(delta_stats_path: str) -> Tuple[float, float]:
    with open(delta_stats_path, "r") as f:
        obj = json.load(f)
    dt = obj.get("dT", obj.get("deltaT", None))
    if dt is None:
        raise ValueError("delta_stats.json must contain key 'dT' with mean/std")
    mean = float(dt.get("mean", 0.0))
    std = float(dt.get("std", 1.0))
    std = max(std, 1e-12)
    return mean, std


def build_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row = edge_index[0]
    col = edge_index[1]
    rel = pos[col] - pos[row]
    dist = torch.norm(rel, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat([rel, dist], dim=1)


def compute_sdf(pos: torch.Tensor, markers_oh: torch.Tensor) -> torch.Tensor:
    boundary_mask = (markers_oh.abs().sum(dim=1) > 0.5)
    bpos = pos[boundary_mask]
    if bpos.numel() == 0:
        return torch.zeros((pos.size(0), 1), dtype=pos.dtype)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(bpos.detach().cpu().numpy())
        d, _ = tree.query(pos.detach().cpu().numpy(), k=1)
        sdf = torch.from_numpy(d).to(pos.dtype).view(-1, 1)
        return sdf
    except Exception:
        pos_cpu = pos.detach().cpu()
        bpos_cpu = bpos.detach().cpu()
        N = pos_cpu.size(0)
        sdf = torch.empty((N,), dtype=pos_cpu.dtype)
        chunk = 1024
        for i in range(0, N, chunk):
            j = min(N, i + chunk)
            d = torch.cdist(pos_cpu[i:j], bpos_cpu)
            sdf[i:j] = d.min(dim=1).values
        return sdf.view(-1, 1)


def build_x(pos: torch.Tensor, markers_oh: torch.Tensor, sdf: torch.Tensor, T: torch.Tensor,
            t_eps: float = 1e-3, t_clip: float = 10.0) -> torch.Tensor:
    Tn = robust_T_norm(T, eps=t_eps, clip=t_clip)
    return torch.cat([pos, markers_oh, Tn.view(-1, 1), sdf], dim=1)


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_cfg = ckpt.get("train_cfg", {})

    hidden_dim = int(train_cfg.get("hidden_dim", 128))
    num_blocks = int(train_cfg.get("num_blocks", 15))
    mlp_layers = int(train_cfg.get("mlp_layers", 2))

    mcfg_kwargs = dict(
        node_in_dim=8,
        edge_in_dim=4,
        hidden_dim=hidden_dim,
        num_processor_blocks=num_blocks,
        mlp_layers=mlp_layers,
        out_dim=1,
    )
    if hasattr(MeshGraphNetConfig, "__dataclass_fields__") and "use_checkpoint" in MeshGraphNetConfig.__dataclass_fields__:
        mcfg_kwargs["use_checkpoint"] = bool(train_cfg.get("use_checkpoint", True))

    model = MeshGraphNetDelta(MeshGraphNetConfig(**mcfg_kwargs))
    state = ckpt.get("model_state", ckpt.get("model_state_raw"))
    if state is None:
        raise ValueError("Checkpoint has no model_state/model_state_raw")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def rollout_autoreg(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    dT_mean: float,
    dT_std: float,
    device: torch.device,
    t_eps: float,
    t_clip: float,
    dt_clip_norm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = sample["pos"].to(device)
    edge_index = sample["edge_index"].to(device)
    markers_oh = sample["markers_oh"].to(device)
    temps = sample["temps"].to(device)  # [T, N]
    times = sample.get("times", torch.arange(temps.size(0), device="cpu", dtype=torch.float32)).to("cpu")

    T_steps, N = temps.size(0), temps.size(1)

    sdf = compute_sdf(pos, markers_oh).to(device)
    edge_attr = build_edge_attr(pos, edge_index)

    T_pred = temps[0].clone()
    pred = torch.empty((T_steps, N), dtype=temps.dtype, device="cpu")
    pred[0] = T_pred.detach().cpu()

    for t in range(T_steps - 1):
        x = build_x(pos, markers_oh, sdf, T_pred, t_eps=t_eps, t_clip=t_clip)
        dT_norm = model(x, edge_index, edge_attr).view(-1)

        if not torch.isfinite(dT_norm).all():
            raise RuntimeError(f"Non-finite dT_norm at step {t}")

        dT_norm = dT_norm.clamp(-dt_clip_norm, dt_clip_norm)
        dT = dT_norm * dT_std + dT_mean
        T_pred = T_pred + dT

        if not torch.isfinite(T_pred).all():
            raise RuntimeError(f"Non-finite temperature at step {t+1}")

        pred[t + 1] = T_pred.detach().cpu()

    return pred.numpy(), times.numpy()


def make_volume_from_points(points: np.ndarray, alpha: float) -> pv.UnstructuredGrid:
    cloud = pv.PolyData(points)
    # Delaunay 3D gives tetra volume for thresholding
    vol = cloud.delaunay_3d(alpha=alpha)
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to one .pt file in data_pyg/")
    ap.add_argument("--ckpt", required=True, help="Path to weights/*_best.pt")
    ap.add_argument("--delta_stats", default="data/index/delta_stats.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["pred", "gt"], default="pred", help="Visualize predicted rollout or ground-truth temps")
    ap.add_argument("--tsolidus", type=float, default=1200.0, help="Solidus temperature threshold")
    ap.add_argument("--alpha", type=float, default=0.0, help="Delaunay3D alpha (0=auto). Try 0..2")
    ap.add_argument("--shell_opacity", type=float, default=0.20)
    ap.add_argument("--t_eps", type=float, default=1e-3)
    ap.add_argument("--t_clip", type=float, default=10.0)
    ap.add_argument("--dt_clip_norm", type=float, default=10.0)
    ap.add_argument("--point_size", type=float, default=6.0, help="Fallback point size if Delaunay fails")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    sample = torch.load(args.pt, map_location="cpu")

    pos = sample["pos"].numpy()
    temps_gt = sample["temps"].numpy()  # [T, N]
    times = sample.get("times", torch.arange(temps_gt.shape[0])).numpy()

    if args.mode == "pred":
        dT_mean, dT_std = load_delta_stats(args.delta_stats)
        model = load_model_from_ckpt(args.ckpt, device)
        temps, times = rollout_autoreg(
            model, sample, dT_mean, dT_std, device,
            t_eps=args.t_eps, t_clip=args.t_clip, dt_clip_norm=args.dt_clip_norm
        )
    else:
        temps = temps_gt

    # Build volume once
    vol = None
    delaunay_ok = True
    try:
        vol = make_volume_from_points(pos, alpha=args.alpha)
    except Exception as e:
        print(f"[warn] Delaunay3D failed, falling back to point cloud rendering. Reason: {e}")
        delaunay_ok = False

    # Compute global scalar range for stable colormap
    tmin = float(np.nanmin(temps))
    tmax = float(np.nanmax(temps))

    plotter = pv.Plotter()
    plotter.add_axes()  # bottom-left axes
    plotter.set_background("white")

    # Actors we will replace on slider callback
    actors = {"solid": None, "liquid": None, "points": None}

    def render_frame(frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, temps.shape[0] - 1))
        T = temps[frame_idx].astype(np.float32)

        # Remove old actors
        for k in list(actors.keys()):
            if actors[k] is not None:
                try:
                    plotter.remove_actor(actors[k])
                except Exception:
                    pass
                actors[k] = None

        if delaunay_ok and vol is not None:
            grid = vol.copy(deep=True)
            grid.point_data["T"] = T

            # Solid (T <= Tsolidus) - translucent
            solid = grid.threshold(value=args.tsolidus, scalars="T", invert=True)
            # Use surface for nicer look
            solid_surf = solid.extract_geometry()

            # Liquid (T > Tsolidus) - opaque core
            liquid = grid.threshold(value=args.tsolidus, scalars="T", invert=False)
            liquid_surf = liquid.extract_geometry()

            actors["solid"] = plotter.add_mesh(
                solid_surf,
                scalars="T",
                clim=(tmin, tmax),
                opacity=args.shell_opacity,
                show_scalar_bar=False,
                smooth_shading=True,
            )
            actors["liquid"] = plotter.add_mesh(
                liquid_surf,
                scalars="T",
                clim=(tmin, tmax),
                opacity=1.0,
                show_scalar_bar=False,
                smooth_shading=True,
            )
        else:
            # Fallback: points colored by T
            cloud = pv.PolyData(pos)
            cloud.point_data["T"] = T
            actors["points"] = plotter.add_mesh(
                cloud,
                scalars="T",
                clim=(tmin, tmax),
                render_points_as_spheres=True,
                point_size=args.point_size,
                opacity=1.0,
                show_scalar_bar=False,
            )

        plotter.add_text(
            f"frame {frame_idx}/{temps.shape[0]-1}   time={float(times[frame_idx]):.4f}",
            position="upper_left",
            font_size=12,
            color="black",
            name="hud",
        )
        plotter.render()

    # Scalar bar (right)
    plotter.add_scalar_bar(
        title="Temperature",
        vertical=True,
        position_x=0.88,
        position_y=0.10,
        height=0.80,
        width=0.08,
        n_labels=5,
        fmt="%.0f",
        color="black",
    )

    # Slider (bottom-left, offset from axes)
    def slider_cb(value):
        render_frame(int(round(value)))

    plotter.add_slider_widget(
        slider_cb,
        rng=[0, temps.shape[0] - 1],
        value=0,
        title="time step",
        pointa=(0.12, 0.06),
        pointb=(0.55, 0.06),
        style="modern",
    )

    render_frame(0)
    plotter.show()


if __name__ == "__main__":
    main()
