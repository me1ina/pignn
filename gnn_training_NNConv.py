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

logging.basicConfig(
    filename='training.log',
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
epochs_warmup = 16
warmup_lr = 1e-3
warmup_patience = 2
epochs_data_loss = 140
data_loss_lr = 1e-4
data_loss_patience = 2
ckpt_epochs = 5
validation_epochs = 4
steps_per_epoch = 2000
num_workers = 2  # Number of workers for DataLoader
stim_scale = 1/0.0066
alpha_for_weights = 1.5

logging.info("=== EXPERIMENT CONFIGURATION ===")
logging.info(f"in_feats          : {in_feats}")
logging.info(f"hidden_feats      : {hidden_feats}")
logging.info(f"out_feats         : {out_feats}")
logging.info(f"edge_feat_dim     : {edge_feat_dim}")
logging.info(f"fanouts           : {fanouts}")
logging.info(f"batch_size        : {batch_size}")
logging.info(f"epochs_warmup     : {epochs_warmup}")
logging.info(f"warmup_lr         : {warmup_lr}")
logging.info(f"warmup_patience   : {warmup_patience}")
logging.info(f"epochs_data_loss  : {epochs_data_loss}")
logging.info(f"data_loss_lr      : {data_loss_lr}")
logging.info(f"data_loss_patience: {data_loss_patience}")
logging.info(f"checkpoint_epochs : {ckpt_epochs}")
logging.info(f"validation_epochs : {validation_epochs}")
logging.info(f"steps_per_epoch   : {steps_per_epoch}")
logging.info(f"num_workers       : {num_workers}")
logging.info(f"stim_scale        : {stim_scale}")
logging.info(f"alpha_for_weights : {alpha_for_weights}")
logging.info("================================\n")

class EdgeAwareGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, edge_feat_dim=1, aggregator='sum'):
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

def save_ckpt(model, feat_mean, feat_std, best_val: bool = False, path: str = "checkpoints/"):
    if best_val:
        ckpt_name = f"checkpoint_best.pth"
    else:
        ckpt_name = f"checkpoint_epoch_last.pth"
    path = path + ckpt_name
    torch.save({
    "model_state": model.state_dict(),
    "feat_mean": feat_mean.cpu(),
    "feat_std": feat_std.cpu(),
    }, path)
    logging.info(f"Checkpoint saved to {path}")

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

model = EdgeAwareGNN(in_feats, hidden_feats, out_feats).to(device)

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

g.ndata['feat'] = g.ndata['feat'][:, 0:6] # 7th feature would be volume which is not needed yet
feat_mean = g.ndata['feat'].mean(0)
feat_std = g.ndata['feat'].std(0).clamp_min(1e-12)
g.ndata['feat'] = (g.ndata['feat'] - feat_mean) / feat_std

g.edata['stim'] = g.edata['stim'] * stim_scale

fm = feat_mean.cpu().numpy()
fs = feat_std.cpu().numpy()
logging.info(f"Feature mean: {fm}")
logging.info(f"Feature std: {fs}")

mins = g.ndata['feat'].amin(0).values.numpy()
maxs = g.ndata['feat'].amax(0).values.numpy()
logging.info(f"Normalized feature ranges per-dim: min {mins}, max {maxs}")

for i,name in enumerate(["x","y","z","sigmaxx","sigmayy","sigmazz"]):
    logging.info(f"{name} range: min {float(g.ndata['feat'][:,i].amin())}, "
                 f"max {float(g.ndata['feat'][:,i].amax())}")

logging.info(f"I stim scaled range: min {float(g.edata['stim'].amin())}, "
             f"max {float(g.edata['stim'].amax())}")

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

# Split node IDs into train / validation 
all_nids = torch.arange(g.num_nodes())
perm = torch.randperm(len(all_nids))
split = int(0.01 * len(all_nids))     # 1% for val
val_nids = all_nids[perm[:split]]
train_nids = all_nids[perm[split:]]

# Create two DataLoaders - one for training and one for validation
sampler = NeighborSampler(
    fanouts, 
    prefetch_node_feats=['feat', 'label', 'w_dist', 'w_pot'], 
    prefetch_edge_feats=['stim']
)
train_loader = DataLoader(
    g, train_nids, sampler,
    batch_size=batch_size, shuffle=True, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)
val_loader = DataLoader(
    g, val_nids, sampler,
    batch_size=batch_size, shuffle=False, drop_last=False,
    num_workers=num_workers, persistent_workers=True
)

optimizer_warmup = torch.optim.Adam(model.parameters(), lr=warmup_lr)
scheduler_warmup = ReduceLROnPlateau(optimizer_warmup, mode='min', factor=0.5, patience=warmup_patience)
loss_fn_warmup = nn.L1Loss()

optimizer_data_loss = torch.optim.Adam(model.parameters(), lr=data_loss_lr)
scheduler_data_loss = ReduceLROnPlateau(optimizer_data_loss, mode='min', factor=0.5, patience=data_loss_patience)

print("Graph loaded and dataloader initialized.")
print("Starting warmup training loop...")

for epoch in tqdm(range(epochs_warmup), desc="Warmup"):
    model.train()
    total_train_loss, n_train_batches = 0.0, 0
    # Warmup Training loop
    for step, (input_nodes, output_nodes, blocks) in enumerate(islice(train_loader, steps_per_epoch)):
        blocks = [b.to(device) for b in blocks]
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
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
            for steps, (input_nodes, output_nodes, blocks) in enumerate(islice(val_loader, steps_per_epoch)):
                blocks = [b.to(device) for b in blocks]
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                pred = model(blocks, x)
                loss = loss_fn_warmup(pred, y)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val   = total_val_loss  / max(1, n_val_batches)
        scheduler_warmup.step(avg_val)

    avg_train = total_train_loss / max(1, n_train_batches)

    val_loss_str = f"Val Loss: {avg_val:.4f} " if (epoch + 1) % validation_epochs == 0 else ""
    msg = (f"[Warmup] Epoch {epoch+1}/{epochs_warmup}  "
          f"Train Loss: {avg_train:.4f}  "
          f"{val_loss_str}"
          f"LR: {optimizer_warmup.param_groups[0]['lr']:.2e}")

    print(msg)
    logging.info(msg)

print("Warmup training done, starting data loss training...")
best_val = float("inf")

for epoch in tqdm(range(epochs_data_loss), desc="Data Loss Training"):
    model.train()
    total_train_loss, n_train_batches = 0.0, 0
    # Training loop
    for step, (input_nodes, output_nodes, blocks) in enumerate(islice(train_loader, steps_per_epoch)):
        blocks = [b.to(device) for b in blocks]
        
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        w = blocks[-1].dstdata['w_pot'].unsqueeze(-1).to(x.dtype)
        w = w * (1.0 + blocks[-1].dstdata['w_dist'].unsqueeze(-1).to(x.dtype))
        
        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            pred = model(blocks, x)
            loss = (w * F.l1_loss(pred, y, reduction='none')).mean()

        optimizer_data_loss.zero_grad(set_to_none=True)
        if scaler_data_loss.is_enabled():
            scaler_data_loss.scale(loss).backward()
            scaler_data_loss.step(optimizer_data_loss)
            scaler_data_loss.update()
        else:
            loss.backward()
            optimizer_data_loss.step()
        total_train_loss += loss.item()
        n_train_batches += 1

    # Validation loop
    
    if (epoch + 1) % validation_epochs == 0:
        model.eval() 
        total_val_loss, n_val_batches = 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            for step, (input_nodes, output_nodes, blocks) in enumerate(islice(val_loader, steps_per_epoch)):
                blocks = [b.to(device) for b in blocks]
                
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                w = blocks[-1].dstdata['w_pot'].unsqueeze(-1).to(x.dtype)
                w = w * (1.0 + blocks[-1].dstdata['w_dist'].unsqueeze(-1).to(x.dtype))

                pred = model(blocks, x)

                loss = (w * F.l1_loss(pred, y, reduction='none')).mean()
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val = total_val_loss / max(1, n_val_batches)
        scheduler_data_loss.step(avg_val)

        if avg_val < best_val - 1e-9:
            best_val = avg_val
            logging.info(f"New best validation loss: {best_val:.4f} at epoch {epoch+1}")
            save_ckpt(model, feat_mean, feat_std, True)

    avg_train = total_train_loss / max(1, n_train_batches)

    if (epoch + 1) % ckpt_epochs == 0:
        save_ckpt(model, feat_mean, feat_std, False)

    val_loss_str = f"Val Loss: {avg_val:.4f} " if (epoch + 1) % validation_epochs == 0 else ""
    msg = (f"[DataLoss] Epoch {epoch+1}/{epochs_data_loss} "
          f"Train Loss: {avg_train:.4f} "
          f"{val_loss_str}"
          f"LR: {optimizer_data_loss.param_groups[0]['lr']:.2e}")
    
    print(msg)
    logging.info(msg)

# Save the model
torch.save({
    "model_state": model.state_dict(),
    "feat_mean": feat_mean.cpu(),
    "feat_std": feat_std.cpu(),
}, "trained_gnn_NNConv.pth")

print(f"Training done, model saved as trained_gnn_NNConv.pth")