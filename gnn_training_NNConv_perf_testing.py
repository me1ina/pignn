from itertools import islice
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import logging
from dgl.dataloading import NeighborSampler, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl.nn import NNConv
from tqdm import tqdm
import os
from time import perf_counter


logging.basicConfig(
    filename='training_performance.log',
    filemode='w',           # overwrite on each run
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# Experiment configuration
in_feats = 6
hidden_feats = 64
out_feats = 1
edge_feat_dim = 1
fanouts = [15, 10, 3]
batch_size = 2048
epochs = 5
lr = 1e-3
steps_per_epoch = 100
num_workers = 2  # Number of workers for DataLoader

logging.info("=== EXPERIMENT CONFIGURATION ===")
logging.info(f"in_feats          : {in_feats}")
logging.info(f"hidden_feats      : {hidden_feats}")
logging.info(f"out_feats         : {out_feats}")
logging.info(f"edge_feat_dim     : {edge_feat_dim}")
logging.info(f"fanouts           : {fanouts}")
logging.info(f"batch_size        : {batch_size}")
logging.info(f"steps_per_epoch    : {steps_per_epoch}")
logging.info(f"num_workers        : {num_workers}")
logging.info("================================\n")

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


def sync(): 
    if torch.cuda.is_available(): torch.cuda.synchronize()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = (device.type == "cuda")
gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
cc = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)  # (major, minor)
is_ampere_plus = use_cuda and (cc[0] >= 8)  # Ampere/Hopper/…

if is_ampere_plus:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = EdgeAwareGNN(in_feats, hidden_feats, out_feats).to(device)

# Prefer BF16 if supported (more stable), else fall back to FP16
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler_warmup    = torch.cuda.amp.GradScaler(enabled=not use_bf16)  # no scaling needed for BF16
scaler_data_loss = torch.cuda.amp.GradScaler(enabled=not use_bf16)

logging.info(f"GPU: {gpu_name} CC:{cc}  is_ampere_plus={is_ampere_plus} "
             f"AMP dtype={'bf16' if use_bf16 else 'fp16' if use_cuda else 'cpu'} "
             f"TF32={'on' if is_ampere_plus else 'off'}")

loaded_graphs, _ = dgl.load_graphs("mesh_graph_vol_area.dgl")
g = loaded_graphs[0]

g.ndata['feat'] = g.ndata['feat'][:, 0:6] # 7th feature would be volume which is not needed yet
mean, std = g.ndata['feat'].mean(0), g.ndata['feat'].std(0)
g.ndata['feat'] = (g.ndata['feat'] - mean) / std

y_min, y_max = g.ndata['label'].min(), g.ndata['label'].max()   # e.g. 0.0 and ~4.0
g.ndata['label_mm'] = (g.ndata['label'] - y_min) / (y_max - y_min)


# At inference time: pred_orig = pred_mm * (y_max - y_min) + y_min

# Split node IDs into train / validation 
all_nids = torch.arange(g.num_nodes())
perm = torch.randperm(len(all_nids))
split = int(0.01 * len(all_nids))     # 1% for val
val_nids = all_nids[perm[:split]]
train_nids = all_nids[perm[split:]]

# Create two DataLoaders - one for training and one for validation
sampler = NeighborSampler(
    fanouts, 
    prefetch_node_feats=['feat', 'label_mm', 'label'], 
    prefetch_edge_feats=['stim']
)
train_loader = DataLoader(
    g, train_nids, sampler,
    batch_size=batch_size, shuffle=True, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)

train_loader_simple = DataLoader(
    g, train_nids, sampler,
    batch_size=batch_size, shuffle=True, drop_last=False
)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.L1Loss()

logging.info("Graph loaded and dataloader initialized.")
logging.info("Starting performance improved training loop...")

t_sample = t_move = t_fwd = t_bwd = 0.0
for epoch in tqdm(range(epochs)):
    model.train()

    for step, batch in enumerate(islice(train_loader, steps_per_epoch)):
        
        t0 = perf_counter()
        input_nodes, output_nodes, blocks = batch
        t1 = perf_counter()  # Measure time taken to unpack batch, no sync() because not done by gpu

        blocks = [b.to(device) for b in blocks]
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label_mm']
        sync(); t2 = perf_counter()
        
        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            pred = model(blocks, x)
            loss = loss_fn(pred, y)
        sync(); t3 = perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if scaler_data_loss.is_enabled():
            scaler_data_loss.scale(loss).backward()
            scaler_data_loss.step(optimizer)
            scaler_data_loss.update()
        else:
            loss.backward()
            optimizer.step()
        sync(); t4 = perf_counter()

        t_sample += t1 - t0
        t_move += t2 - t1
        t_fwd += t3 - t2
        t_bwd += t4 - t3

    total_time = t_sample + t_move + t_fwd + t_bwd
    logging.info(f"[TIMING] sampling {t_sample/total_time:.1%} | move {t_move/total_time:.1%} | fwd {t_fwd/total_time:.1%} | bwd {t_bwd/total_time:.1%}")

logging.info("Starting simple training loop...")
t_sample = t_move = t_fwd = t_bwd = 0.0
for epoch in tqdm(range(epochs)):
    model.train()

    for step, batch in enumerate(islice(train_loader_simple, steps_per_epoch)):
        
        t0 = perf_counter()
        input_nodes, output_nodes, blocks = batch
        t1 = perf_counter()  # Measure time taken to unpack batch, no sync() because not done by gpu

        blocks = [b.to(device) for b in blocks]
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label_mm']
        sync(); t2 = perf_counter()
        
        pred = model(blocks, x)
        loss = loss_fn(pred, y)
        sync(); t3 = perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sync(); t4 = perf_counter()

        t_sample += t1 - t0
        t_move += t2 - t1
        t_fwd += t3 - t2
        t_bwd += t4 - t3

    total_time = t_sample + t_move + t_fwd + t_bwd
    logging.info(f"[TIMING] sampling {t_sample/total_time:.1%} | move {t_move/total_time:.1%} | fwd {t_fwd/total_time:.1%} | bwd {t_bwd/total_time:.1%}")
