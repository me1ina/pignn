from itertools import islice
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import logging
from dgl.dataloading import NeighborSampler, DataLoader, ClusterGCNSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl.nn import NNConv
from tqdm import tqdm
import os

logging.basicConfig(
    filename='training_laplace.log',
    filemode='w',           # overwrite on each run
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# Experiment configuration
in_feats = 6
hidden_feats = 64
out_feats = 1
edge_feat_dim = 2
fanouts = [15, 10, 3]
batch_size = 2048
epochs_warmup = 0
warmup_lr = 1e-3
warmup_patience = 2
epochs_main = 2
main_lr = 1e-4
main_patience = 3
ckpt_epochs = 5
validation_epochs = 4
steps_per_epoch = 2000
num_workers = 2  # Number of workers for DataLoader
stim_scale = 1/(0.0066 *2) # map µA to ~[0,1]
alpha_for_weights = 2
coord_max = 35.0 #mm
z_center = 17.5 #mm
sigma_max = 2.0 # S/m

logging.info("=== EXPERIMENT CONFIGURATION ===")
logging.info(f"in_feats                 : {in_feats}")
logging.info(f"hidden_feats             : {hidden_feats}")
logging.info(f"out_feats                : {out_feats}")
logging.info(f"edge_feat_dim            : {edge_feat_dim}")
logging.info(f"fanouts                  : {fanouts}")
logging.info(f"batch_size               : {batch_size}")
logging.info(f"epochs_warmup            : {epochs_warmup}")
logging.info(f"warmup_lr                : {warmup_lr}")
logging.info(f"warmup_patience          : {warmup_patience}")
logging.info(f"epochs_main              : {epochs_main}")
logging.info(f"main_lr                  : {main_lr}")
logging.info(f"main_patience            : {main_patience}")
logging.info(f"checkpoint_epochs        : {ckpt_epochs}")
logging.info(f"validation_epochs        : {validation_epochs}")
logging.info(f"steps_per_epoch          : {steps_per_epoch}")
logging.info(f"num_workers              : {num_workers}")
logging.info(f"stim_scale               : {stim_scale}")
logging.info(f"alpha_for_weights        : {alpha_for_weights}")
logging.info("================================\n")

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
        e1 = edge_feats(blocks[0])
        h = F.relu(self.norm1(self.conv1(blocks[0], x, e1)))
        # Layer 2
        e2 = edge_feats(blocks[1])
        h = F.relu(self.norm2(self.conv2(blocks[1], h, e2)))
        # Layer 3
        e3 = edge_feats(blocks[2])
        return F.softplus(self.conv3(blocks[2], h, e3))
    
    def forward_full(self, g, x):
        e = g.edata['stim'].unsqueeze(-1)
        h = F.relu(self.norm1(self.conv1(g, x, e)))
        h = F.relu(self.norm2(self.conv2(g, h, e)))   # same g each layer
        out = self.conv3(g, h, e)
        return F.softplus(out)

def save_ckpt(model, best_val: bool = False, path: str = "checkpoints/"):
    if best_val:
        ckpt_name = f"checkpoint_best.pth"
    else:
        ckpt_name = f"checkpoint_epoch_last.pth"
    path = path + ckpt_name
    torch.save({
    "model_state": model.state_dict()
    }, path)
    logging.info(f"Checkpoint saved to {path}")

def get_stim_center(g):
    eids = (g.edata['stim'] != 0).nonzero(as_tuple=False).reshape(-1)
    u, v = g.find_edges(eids)
    stim_nodes = torch.unique(torch.cat([u, v]))
    return g.ndata['feat'][stim_nodes, :3].mean(0)  # xyz in mm

def norm_feats(feats, stim_center):
    x = torch.empty_like(feats)
    x[:, 0:3] = (feats[:, 0:3] - stim_center.to(feats.device)) / coord_max
    x[:, 3:6] = (feats[:, 3:6]).clamp_min(0.0) / sigma_max
    return x

def laplace_physics_loss_graph(graph, potential):
  # Edge endpoints in *local IDs*
    src, dst = graph.edges()

    coords = graph.ndata['feat'][:, 0:3]
    sigma  = graph.srcdata['feat'][:, 3:6]
    I_stim   = graph.edata['stim'].view(-1, 1)
    face_areas   = graph.edata['face_area'].view(-1, 1)

    # Map to local node features
    pot_src, pot_dst = potential[src], potential[dst]
    delta_V = pot_src - pot_dst

    delta_x = coords[src] - coords[dst]
    dist = torch.norm(delta_x, dim=1, keepdim=True)
    dist_sq = dist ** 2

    sigma_eff = sigma[src] * ((delta_x * 1e-3) ** 2)
    sigma_eff = sigma_eff.sum(dim=1, keepdim=True) / (dist_sq * 1e-6 + 1e-12)

    flux_density = sigma_eff * delta_V / (dist + 1e-12)
    flux_current = flux_density * face_areas

    zero_flux = torch.zeros_like(potential)
    inflow = zero_flux.index_add(0, dst, flux_current)
    outflow = zero_flux.index_add(0, src, flux_current)

    stim_per_cell = torch.zeros_like(potential)
    stim_per_cell = stim_per_cell.index_add(0, dst, 0.25 * I_stim)
    stim_per_cell = stim_per_cell.index_add(0, src, 0.25 * I_stim)

    divergence = inflow - outflow
    residual = divergence - stim_per_cell

    return (residual.abs()).mean()

print("Start training process")
print("Loading graph and initializing model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = (device.type == "cuda")
gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
cc = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)  # (major, minor)
is_ampere_plus = use_cuda and (cc[0] >= 8)  # Ampere/Hopper/…
print(f"GPU: {gpu_name}  CC:{cc}")

if is_ampere_plus:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = EdgeAwareGNN(in_feats, hidden_feats, out_feats, edge_feat_dim).to(device)

# Prefer BF16 if supported (more stable), else fall back to FP16
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler_warmup    = torch.cuda.amp.GradScaler(enabled=not use_bf16)  # no scaling needed for BF16
scaler_data_loss = torch.cuda.amp.GradScaler(enabled=not use_bf16)

logging.info(f"GPU: {gpu_name} CC:{cc}  is_ampere_plus={is_ampere_plus}\n"
             f"AMP dtype={'bf16' if use_bf16 else 'fp16' if use_cuda else 'cpu'}\n"
             f"TF32={'on' if is_ampere_plus else 'off'}\n")

loaded_graphs, _ = dgl.load_graphs("mesh_graph_vol_area.dgl")
g = loaded_graphs[0]
#g = dgl.to_bidirected(g, copy_ndata=True)

#create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)
logging.info(f"Checkpoints are saved in {os.path.abspath('checkpoints')}")

g.ndata['feat'] = g.ndata['feat'][:, 0:6].to(torch.float32) # 7th feature would be volume which is not needed yet

for i,name in enumerate(["x","y","z","sigmaxx","sigmayy","sigmazz"]):
    min_val = float(g.ndata['feat'][:,i].amin())
    max_val = float(g.ndata['feat'][:,i].amax())
    logging.info(f"{name} range: min {min_val}, max {max_val}")
    if i < 2:
        min_val = min_val/coord_max
        max_val = max_val/coord_max
    elif i == 2:
        min_val = (min_val - z_center)/z_center
        max_val = (max_val - z_center)/z_center
    else:
        min_val = min_val/sigma_max
        max_val = max_val/sigma_max
    logging.info(f"{name} scaled range: min {min_val}, max {max_val}")

logging.info(f"I stim range: min {float(g.edata['stim'].amin())}, "
             f"max {float(g.edata['stim'].amax())}")

logging.info(f"I stim range scaled: min {float(g.edata['stim'].amin()) * stim_scale}, "
             f"max {float(g.edata['stim'].amax()) * stim_scale}")

logging.info(f"Potential range: min {float(g.ndata['label'].amin())}, "
             f"max {float(g.ndata['label'].amax())}")

#add distance-to-stim weight
stim_mask = (g.edata['stim'] != 0)
if stim_mask.ndim > 1:
    stim_mask = stim_mask.squeeze(-1)
eids = g.edges(form='eid')[stim_mask]
stim_src, stim_dst = g.find_edges(eids)
stim_nodes = torch.unique(torch.cat([stim_src, stim_dst]))
_, khop_nodes = dgl.khop_in_subgraph(g, stim_nodes, k=3)
w_dist = torch.zeros(g.num_nodes(), dtype=torch.float32)
w_dist[khop_nodes] = 1.0
g.ndata['w_dist'] = w_dist

g.ndata['w_pot'] = 1 + alpha_for_weights * (g.ndata['label'] / g.ndata['label'].max())

stim_center = get_stim_center(g)

# Split node IDs into train / validation 
all_nids = torch.arange(g.num_nodes())
perm = torch.randperm(len(all_nids))
split = int(0.05 * len(all_nids))     # 5% for val
val_nids = all_nids[perm[:split]]
train_nids = all_nids[perm[split:]]

# Create two DataLoaders - one for training and one for validation
warmup_sampler = NeighborSampler(
    fanouts, 
    prefetch_node_feats=['feat', 'label'], 
    prefetch_edge_feats=['stim']
)
data_sampler = ClusterGCNSampler(
    g, k=1500,                 # number of clusters (tune)
    prefetch_node_feats=['feat','label', 'w_dist', 'w_pot'],
    prefetch_edge_feats=['stim','face_area'],
)

warmup_train_loader = DataLoader(
    g, train_nids, warmup_sampler,
    batch_size=batch_size, shuffle=True, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)
warmup_val_loader = DataLoader(
    g, val_nids, warmup_sampler,
    batch_size=batch_size, shuffle=False, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)
data_train_loader = DataLoader(
    g, train_nids, data_sampler,
    batch_size=batch_size, shuffle=True, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)

data_val_loader = DataLoader(
    g, val_nids, data_sampler,
    batch_size=batch_size, shuffle=True, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)

optimizer_warmup = torch.optim.Adam(model.parameters(), lr=warmup_lr)
scheduler_warmup = ReduceLROnPlateau(optimizer_warmup, mode='min', factor=0.5, patience=warmup_patience)
loss_fn_warmup = nn.L1Loss()

optimizer_data_loss = torch.optim.Adam(model.parameters(), lr=main_lr)
scheduler_data_loss = ReduceLROnPlateau(optimizer_data_loss, mode='min', factor=0.1, patience=main_patience)

print("Graph loaded and dataloader initialized.")
print("Starting warmup training loop...")

for epoch in tqdm(range(epochs_warmup), desc="Warmup"):
    model.train()
    total_train_loss, n_train_batches = 0.0, 0
    # Warmup Training loop
    for step, (input_nodes, output_nodes, blocks) in enumerate(islice(warmup_train_loader, steps_per_epoch)):
        blocks = [b.to(device) for b in blocks]
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        
        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            x = norm_feats(x, stim_center)
            pred = model(blocks, x)
            loss = loss_fn_warmup(pred, y)

        optimizer_warmup.zero_grad(set_to_none=True)
        if scaler_warmup.is_enabled():
            scaler_warmup.scale(loss).backward()
            scaler_warmup.step(optimizer_warmup)
            scaler_warmup.update()
        else:
            loss.backward()
            optimizer_warmup.step()
        total_train_loss += loss.item()
        n_train_batches += 1

    # Warmup Validation loop
    if (epoch + 1) % validation_epochs == 0:
        model.eval()
        total_val_loss, n_val_batches = 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            for steps, (input_nodes, output_nodes, blocks) in enumerate(islice(warmup_val_loader, steps_per_epoch)):
                blocks = [b.to(device) for b in blocks]
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                x = norm_feats(x, stim_center)
                pred = model(blocks, x)
                loss = loss_fn_warmup(pred, y)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val   = total_val_loss  / max(1, n_val_batches)
        scheduler_warmup.step(avg_val)

    avg_train = total_train_loss / max(1, n_train_batches)

    val_loss_str = f"Val Loss: {avg_val:.6f} " if (epoch + 1) % validation_epochs == 0 else ""
    msg = (f"[Warmup] Epoch {epoch+1}/{epochs_warmup}  "
          f"Train Loss: {avg_train:.6f}  "
          f"{val_loss_str}"
          f"LR: {optimizer_warmup.param_groups[0]['lr']:.2e}")

    print(msg)
    logging.info(msg)

print("Warmup training done, starting data loss training...")
best_val = float("inf")

for epoch in tqdm(range(epochs_main), desc="Data Loss Training"):
    model.train()
    total_train_loss, total_data_loss, total_phys_loss, n_train_batches = 0.0, 0.0, 0.0, 0
    # Training loop
    for step, batch in enumerate(islice(data_train_loader, steps_per_epoch)):
        batch = batch.to(device)
        
        x = batch.ndata['feat']
        y = batch.ndata['label']
        w = batch.ndata['w_pot'].unsqueeze(-1).to(x.dtype)
        w = w * (1.0 + batch.ndata['w_dist'].unsqueeze(-1).to(x.dtype))

        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            x = norm_feats(x, stim_center)
            pred = model.forward_full(batch, x)
            data_loss = (w * F.l1_loss(pred, y, reduction='none')).mean()
            phys_loss = laplace_physics_loss_graph(batch, pred)
            loss = data_loss + phys_loss

        optimizer_data_loss.zero_grad(set_to_none=True)
        if scaler_data_loss.is_enabled():
            scaler_data_loss.scale(loss).backward()
            scaler_data_loss.step(optimizer_data_loss)
            scaler_data_loss.update()
        else:
            loss.backward()
            optimizer_data_loss.step()
        total_train_loss += loss.item()
        total_data_loss += data_loss.item()
        total_phys_loss += phys_loss.item()
        n_train_batches += 1

    # Validation loop
    if (epoch + 1) % validation_epochs == 0:
        model.eval() 
        total_val_loss, total_data_val_loss, total_phys_val_loss,n_val_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            for step, batch in enumerate(islice(data_val_loader, steps_per_epoch)):
                batch = batch.to(device)

                x = batch.ndata['feat']
                y = batch.ndata['label']
                w = batch.ndata['w_pot'].unsqueeze(-1).to(x.dtype)
                w = w * (1.0 + batch.ndata['w_dist'].unsqueeze(-1).to(x.dtype))
                x = norm_feats(x, stim_center)
                pred = model.forward_full(batch, x)

                data_loss = (w * F.l1_loss(pred, y, reduction='none')).mean()
                phys_loss = laplace_physics_loss_graph(batch, pred)
                loss = data_loss + phys_loss
                total_val_loss += loss.item()
                total_data_val_loss += data_loss.item()
                total_phys_val_loss += phys_loss.item()
                n_val_batches += 1

        avg_total_val = total_val_loss / max(1, n_val_batches)
        avg_data_val  = total_data_val_loss / max(1, n_val_batches)
        avg_phys_val  = total_phys_val_loss / max(1, n_val_batches)
        scheduler_data_loss.step(avg_total_val)

        if avg_total_val < best_val - 1e-9:
            best_val = avg_total_val
            logging.info(f"New best validation loss: {best_val:.8f} at epoch {epoch+1}")
            save_ckpt(model, True)

    avg_total_train = total_train_loss / max(1, n_train_batches)
    avg_total_data = total_data_loss / max(1, n_train_batches)
    avg_total_physics = total_phys_loss / max(1, n_train_batches)

    if (epoch + 1) % ckpt_epochs == 0:
        save_ckpt(model, False)

    val_loss_str = f"\nTotal Val Loss: {avg_total_val:.10f} Data Val Loss: {avg_data_val:.10f} Physics Val Loss: {avg_phys_val:.10f}" if (epoch + 1) % validation_epochs == 0 else ""
    msg = (f"[DataLoss] Epoch {epoch+1}/{epochs_main} "
          f"Train Loss: {avg_total_train:.10f} "
          f"Data Loss: {avg_total_data:.10f} "
          f"Physics Loss: {avg_total_physics:.10f} "
          f"LR: {optimizer_data_loss.param_groups[0]['lr']:.2e}"
          f"{val_loss_str}")
    
    print(msg)
    logging.info(msg)

# Save the model
torch.save({
    "model_state": model.state_dict(),
}, "trained_gnn_NNConv_laplace_v3.pth")

print(f"Training done, model saved as trained_gnn_NNConv_laplace_v3.pth")