import os
import torch
import meshio
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# Settings
RAW_DIR = "./data"
PROC_DIR = "./data_pyg"
os.makedirs(PROC_DIR, exist_ok=True)

all_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.npz')])

print("Converting raw data to PyG binary format...")
for f_name in tqdm(all_files):
    base_name = f_name.replace('.npz', '')
    npz_path = os.path.join(RAW_DIR, f_name)
    msh_path = os.path.join(RAW_DIR, base_name + '.msh')

    # 1. Read mesh (once per file)
    mesh = meshio.read(msh_path)
    pos = torch.from_numpy(mesh.points).float() / 1000.0
    cells = mesh.cells_dict.get('tetra', mesh.cells_dict.get('tetra4'))
    edges = np.concatenate([cells[:, [0, 1]], cells[:, [0, 2]], cells[:, [0, 3]],
                            cells[:, [1, 2]], cells[:, [1, 3]], cells[:, [2, 3]]], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edge_index = torch.from_numpy(np.concatenate([edges, edges[:, [1, 0]]], axis=0).T).long()

    markers = torch.from_numpy(mesh.point_data['markers']).long() if 'markers' in mesh.point_data else torch.zeros(pos.size(0), dtype=torch.long)
    markers_oh = torch.nn.functional.one_hot(markers, num_classes=3).float()

    # 2. Read all temperature snapshots
    data_npz = np.load(npz_path)
    temps = torch.from_numpy(data_npz['snapshots_nodal']).float() # (T, N)
    times = torch.from_numpy(data_npz['times']).float() # (T,)

    # 3. Save compact object
    # We save geometry and ALL time steps into a single .pt file
    payload = {
        'pos': pos,
        'edge_index': edge_index,
        'markers_oh': markers_oh,
        'temps': temps,
        'times': times
    }
    torch.save(payload, os.path.join(PROC_DIR, f"{base_name}.pt"))
