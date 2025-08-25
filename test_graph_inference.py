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


class EdgeAwareGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, edge_feat_dim=1, aggregator='mean'):
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
        e1 = blocks[0].edata['stim'].unsqueeze(-1)  # shape [E0,1]
        h = F.relu(self.norm1(self.conv1(blocks[0], x, e1)))
        # Layer 2
        e2 = blocks[1].edata['stim'].unsqueeze(-1)  # shape [E1,1]
        h = F.relu(self.norm2(self.conv2(blocks[1], h, e2)))
        # Layer 3
        e3 = blocks[2].edata['stim'].unsqueeze(-1)  # shape [E2,1]
        return F.softplus(self.conv3(blocks[2], h, e3))

def perform_inference():
    inference_graph_name = "mesh_graph_vol_area.dgl"
    model_name = "trained_gnn_data_loss_test_NNConv.pth"#"trained_gnn_combi_loss_seperated_test.pth" 

    in_feats = 6
    hidden_feats = 64
    out_feats = 1
    edge_feat_dim = 1
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
    g = g.to(device)

    #raw_coords = g.ndata['feat'][:, :3].clone()

    nids = torch.arange(g.num_nodes())
    sampler = NeighborSampler(
        list(fanouts),
        prefetch_node_feats=['feat'],
        prefetch_edge_feats=['stim']
    )
    loader = DataLoader(
        g, nids, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True
    )

    model = EdgeAwareGNN(in_feats, hidden_feats, out_feats).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    print("Graph loaded, starting inference...")
    start_time = time.time()

    preds = torch.empty((g.num_nodes(), out_feats), dtype=torch.float32)  # store on CPU
    with torch.no_grad():
        autocast_ctx = (torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype)
                        if use_cuda else torch.autocast("cpu", dtype=torch.float32, enabled=False))
        for input_nodes, output_nodes, blocks in tqdm(loader, desc="Inference"):
            blocks = [b.to(device) for b in blocks]
            x_b = blocks[0].srcdata['feat'][:, :in_feats]
            with autocast_ctx:
                y_b = model(blocks, x_b)  # [N_dst, 1]
            preds[output_nodes] = y_b.detach().float().cpu()

    g.ndata["Electric_potential"] = preds.squeeze(1)

    end_time = time.time() - start_time
    logging.info(f"Inference completed in {end_time:.3f} seconds, storing results in graph...")

    dgl.save_graphs("inference_gnn_test_VagusA1924_HC0_AS1.dgl", [g])
    logging.info("Graph saved to inference_gnn_test_VagusA1924_HC0_AS1.dgl")

    return g

def visualize_graph(g):
    print("Visualizing results as point cloud...")

    node_points = g.ndata["feat"][:, :3].numpy()  # coords
    point_cloud = pv.PolyData(node_points)

    #points where I_stim > 0
    I_stim_mask = (g.edata["stim"] > 0).squeeze()
    stimulated_eids = g.edges(form='eid')[I_stim_mask]

    stimulated_nodes = g.find_edges(stimulated_eids)  # get source nodes of stimulated edges

    stimulated_points = g.ndata["feat"][stimulated_nodes[0]][:, :3].cpu().numpy()  # coords of stimulated nodes
    point_cloud_stimulated = pv.PolyData(stimulated_points)

    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars=g.ndata["Electric_potential"].cpu().numpy(), point_size=5) # Electric_potential
    plotter.add_mesh(point_cloud_stimulated, color='yellow', point_size=10, render_points_as_spheres=True)
    plotter.show()

g = perform_inference()
visualize_graph(g)
