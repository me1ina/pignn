import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import NNConv
from dgl.dataloading import NeighborSampler, DataLoader
import pyvista as pv
from pyvista import CellType
import numpy as np
import time
from tqdm import tqdm
import logging

coord_max = 35.0     # mm (x,y in [-33,33], z in [0,35])
z_center = 17.5     # mm
sigma_max = 2.0      # S/m
stim_scale = 1.0 / (0.0066 * 2)   # maps ~0..0.0066 µA -> ~0..1 (or your final chosen value)

def edge_feats(b):
    s = (b.edata['stim'] * stim_scale).unsqueeze(-1)
    ones = torch.ones_like(s)
    return torch.cat([s, ones], dim=-1) 
class EdgeAwareGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, edge_feat_dim, aggregator='mean'):
        super().__init__()
        # Layer 1: in_feats → hidden_feats
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_feat_dim, in_feats * hidden_feats),
            nn.ReLU(),
            nn.Linear(in_feats * hidden_feats, in_feats * hidden_feats),
        )
        self.conv1 = NNConv(in_feats, hidden_feats, self.edge_mlp1, aggregator_type=aggregator)
        self.norm1 = nn.LayerNorm(hidden_feats)
        # Layer 2: hidden_feats → hidden_feats
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_feats * hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats * hidden_feats, hidden_feats * hidden_feats),
        )
        self.conv2 = NNConv(hidden_feats, hidden_feats, self.edge_mlp2, aggregator_type=aggregator)
        self.norm2 = nn.LayerNorm(hidden_feats)
        # Layer 3: hidden_feats → out_feats
        self.edge_mlp3 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_feats * out_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats * out_feats, hidden_feats * out_feats),
        )
        self.conv3 = NNConv(hidden_feats, out_feats, self.edge_mlp3, aggregator_type=aggregator)

    def forward(self, blocks, x):
        """
        blocks: list of 3 DGLBlock objects
        x:     input node features for the src nodes of blocks[0]
        """
        # Layer 1
        e1 = edge_feats(blocks[0])  # shape [E0,2]
        h = F.relu(self.norm1(self.conv1(blocks[0], x, e1)))
        # Layer 2
        e2 = edge_feats(blocks[1])  # shape [E1,2]
        h = F.relu(self.norm2(self.conv2(blocks[1], h, e2)))
        # Layer 3
        e3 = edge_feats(blocks[2])  # shape [E2,2]
        return F.softplus(self.conv3(blocks[2], h, e3))
    

def get_stim_center(g):
    eids = (g.edata['stim'] != 0).nonzero(as_tuple=False).reshape(-1)
    u, v = g.find_edges(eids)
    stim_nodes = torch.unique(torch.cat([u, v]))
    return g.ndata['feat'][stim_nodes, :3].mean(0)  # xyz in mm

def norm_feats(feats, stim_center):
    feats[:, 0:3] = (feats[:, 0:3] - stim_center.to(feats.device)) / coord_max 
    #feats[:, 0] = feats[:, 0] / coord_max               # x ~ [-1,1]
    #feats[:, 1] = feats[:, 1] / coord_max               # y ~ [-1,1]
    #feats[:, 2] = (feats[:, 2] - z_center) / z_center   # z ~ [-1,1]

    # map conductivity to [0,1] (optionally clip tiny floor to reduce skew)
    feats[:, 3:6] = (feats[:, 3:6]).clamp_min(0.0) / sigma_max
    return feats

#inference_graph_name = "graph_area_VagusA6050_HC0_AS1.1.dgl"
inference_graph_name = "graph_area_VagusA1924_HC240_AS1.2.dgl"
#inference_graph_name = "graph_area_VagusAA1924_HT60_AS1.1.dgl"
#inference_graph_name = "graph_area_VagusAA1924_HC0_AS1.7.dgl"
#inference_graph_name = "mesh_graph_vol_area.dgl"
#inference_graph_name = "graph_area_VagusA6050_HC0_AS1.1.dgl"
model_name = "trained_gnn_NNConv_dirichlet_v2.pth"
#model_name = "trained_gnn_NNConv_laplace_v1.pth" 

in_feats = 6
hidden_feats = 64
out_feats = 1
edge_feat_dim = 2
num_workers = 2
fanouts=(15,10,3)
batch_size=2048

logging.basicConfig(
    filename='inference.log',
    filemode='w',           # overwrite on each run
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = (device.type == "cuda")

if use_cuda:
    cc = torch.cuda.get_device_capability(0)
    is_ampere_plus = cc[0] >= 8
    if is_ampere_plus:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
else:
    amp_dtype = torch.float32


loaded_graphs, _ = dgl.load_graphs(inference_graph_name)
g = loaded_graphs[0]
stim_center = get_stim_center(g)


model = EdgeAwareGNN(in_feats, hidden_feats, out_feats, edge_feat_dim).to(device)
ckpt = torch.load(model_name, map_location=device)
model.load_state_dict(ckpt["model_state"])

g.ndata['feat'] = g.ndata['feat'][:, 0:in_feats] # 7th feature would be volume which is not needed yet

nids = torch.arange(g.num_nodes(), dtype=torch.int64)
sampler = NeighborSampler(
    list(fanouts),
    prefetch_node_feats=['feat'],
    prefetch_edge_feats=['stim']
)

print("graph device:", g.device)
print("nids device :", nids.device)

loader = DataLoader(
    g, nids, sampler,
    batch_size=batch_size, shuffle=False, drop_last=False,
    num_workers=num_workers, persistent_workers=(num_workers > 0)
)

model.eval()

print("Graph loaded, starting inference...")
start_time = time.time()

preds = torch.empty((g.num_nodes(), out_feats), dtype=torch.float32)  # store on CPU
with torch.no_grad():
    autocast_ctx = (torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype)
                    if use_cuda else torch.autocast("cpu", dtype=torch.float32, enabled=False))
    for input_nodes, output_nodes, blocks in tqdm(loader, desc="Inference"):
        blocks = [b.to(device) for b in blocks]

        x_b = norm_feats(blocks[0].srcdata['feat'][:, :in_feats], stim_center)
        with autocast_ctx:
            y_b = model(blocks, x_b)  # [N_dst, 1]
        preds[output_nodes] = y_b.detach().float().cpu()

g.ndata["Electric_potential"] = preds.squeeze(1)

end_time = time.time() - start_time
logging.info(f"Inference completed in {end_time:.3f} seconds, storing results in graph...")

dgl.save_graphs("inference_gnn_test_dirichlet_VagusA1924_HC240_AS2.dgl", [g])
logging.info("Graph saved to inference_gnn_test_dirichlet_VagusA1924_HC240_AS2.dgl")

