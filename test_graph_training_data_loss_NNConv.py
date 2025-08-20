import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import logging
from dgl.dataloading import NeighborSampler, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl.nn import NNConv
from tqdm import tqdm

logging.basicConfig(
    filename='training.log',
    filemode='w',           # overwrite on each run
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# Experiment configuration
in_feats = 6
hidden_feats = 128
out_feats = 1
edge_feat_dim = 1
fanouts = [20, 10, 5]
batch_size = 1024
epochs_warmup = 50
warmup_lr = 1e-3
warmup_patience = 5
epochs_data_loss = 250
data_loss_lr = 1e-4
data_loss_patience = 10
alpha_for_weights = 2.0  # Weighting factor for the loss function

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
logging.info(f"weight_alpha      : {alpha_for_weights}")
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

def save_ckpt(path: str = "checkpoints/", *, epoch, model, optimizer, scheduler, best_val,
              y_min, y_max, feat_mean, feat_std, config):
    path = path + f"checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val": best_val,
        # scalers needed for inference/repro
        "y_min": float(y_min),
        "y_max": float(y_max),
        "feat_mean": feat_mean.cpu(),
        "feat_std": feat_std.cpu(),
        "config": config,
        # RNG states help reproducibility on resume
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)

print("Start training process")
print("Loading graph and initializing model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgeAwareGNN(in_feats, hidden_feats, out_feats).to(device)

loaded_graphs, _ = dgl.load_graphs("mesh_graph_vol_area.dgl")
g = loaded_graphs[0]

g.ndata['feat'] = g.ndata['feat'][:, 0:6] # 7th feature would be volume which is not needed yet
mean, std = g.ndata['feat'].mean(0), g.ndata['feat'].std(0)
g.ndata['feat'] = (g.ndata['feat'] - mean) / std

y_min, y_max = g.ndata['label'].min(), g.ndata['label'].max()   # e.g. 0.0 and ~4.0
g.ndata['label_mm'] = (g.ndata['label'] - y_min) / (y_max - y_min)

weights_full = (1 + alpha_for_weights * (g.ndata['label'] / y_max)).to(device)

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
    batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    g, val_nids, sampler,
    batch_size=batch_size, shuffle=False, drop_last=False
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
    total_train_loss = 0
    # Warmup Training loop
    for input_nodes, output_nodes, blocks in train_loader:
        blocks = [b.to(device) for b in blocks]
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label_mm']
        pred = model(blocks, x)
        loss = loss_fn_warmup(pred, y)

        optimizer_warmup.zero_grad()
        loss.backward()
        optimizer_warmup.step()
        total_train_loss += loss.item()

    # Warmup Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in val_loader:
            blocks = [b.to(device) for b in blocks]
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label_mm']
            pred = model(blocks, x)
            loss = loss_fn_warmup(pred, y)
            total_val_loss += loss.item()
    avg_train = total_train_loss / len(train_loader)
    avg_val   = total_val_loss   / len(val_loader)

    scheduler_warmup.step(avg_val)

    msg = (f"[Warmup] Epoch {epoch+1}/{epochs_warmup}  "
          f"Train Loss: {avg_train:.4f}  "
          f"Val Loss: {avg_val:.4f}  "
          f"LR: {optimizer_warmup.param_groups[0]['lr']:.2e}")

    print(msg)
    logging.info(msg)

print("Warmup training done, starting data loss training...")


for epoch in tqdm(range(epochs_data_loss), desc="Data Loss Training"):
    model.train()
    total_train_loss = 0
    # Training loop
    for input_nodes, output_nodes, blocks in train_loader:
        blocks = [b.to(device) for b in blocks]

        w = weights_full[output_nodes].unsqueeze(-1)
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        
        pred = model(blocks, x)

        loss = (w * F.l1_loss(pred, y, reduction='none')).mean()

        optimizer_data_loss.zero_grad()
        loss.backward()
        optimizer_data_loss.step()
        total_train_loss += loss.item()

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in val_loader:
            blocks = [b.to(device) for b in blocks]
                    
            w = weights_full[output_nodes].unsqueeze(-1)
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']

            pred = model(blocks, x)

            loss = (w * F.l1_loss(pred, y, reduction='none')).mean()
            total_val_loss += loss.item()

    avg_train = total_train_loss / len(train_loader)
    avg_val   = total_val_loss   / len(val_loader)

    scheduler_data_loss.step(avg_val)

    msg = (f"[DataLoss] Epoch {epoch+1}/{epochs_data_loss}  "
          f"Train Loss: {avg_train:.4f}  "
          f"Val Loss: {avg_val:.4f}  "
          f"LR: {optimizer_data_loss.param_groups[0]['lr']:.2e}")
    
    print(msg)
    logging.info(msg)

# Save the model
torch.save(model.state_dict(), "trained_gnn_data_loss_test_NNConv.pth")
print("Training done, model saved as trained_gnn_data_loss_test_NNConv.pth")