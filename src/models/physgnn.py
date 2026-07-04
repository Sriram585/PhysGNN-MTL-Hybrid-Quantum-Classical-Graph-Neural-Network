import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
import pytorch_lightning as pl
import pennylane as qml

# Configure the PennyLane Quantum Device
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

class PhysGNN_MTL(pl.LightningModule):
    def __init__(self, node_dim, edge_dim, n_qubits=12, q_depth=2, lr=1e-3, scaler=None):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler'])
        self.lr = lr
        self.scaler = scaler
        
        # --- Classical Graph Module ---
        self.conv1 = TransformerConv(node_dim, 32, edge_dim=edge_dim)
        self.conv2 = TransformerConv(32, 64, edge_dim=edge_dim)
        self.fc_classical = nn.Linear(64, n_qubits)
        
        # --- Quantum Circuit Module ---
        weight_shapes = {"weights": (q_depth, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # --- Multi-Task Regression Heads ---
        self.head_band_gap = nn.Linear(3, 1)
        self.head_form_energy = nn.Linear(3, 1)
        self.head_bulk_mod = nn.Linear(3, 1)
        
        # Tracking losses for analytical plotting
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Classical Graph Feature Extraction
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # Global Pooling
        x_pool = global_mean_pool(x, batch)
        
        # Linear projection exactly to 12 dimensions
        x_proj = self.fc_classical(x_pool)
        
        # Scale to [0, pi] for quantum embedding
        x_q_in = torch.sigmoid(x_proj) * torch.pi
        
        # Quantum Circuit mapping to 3 expectation values
        q_out = self.qlayer(x_q_in) 
        
        # Regression Heads
        bg = self.head_band_gap(q_out)
        fe = self.head_form_energy(q_out)
        bm = self.head_bulk_mod(q_out)
        
        return torch.cat([bg, fe, bm], dim=1) 

    def custom_loss(self, preds, targets):
        bg_pred, fe_pred, bm_pred = preds[:, 0], preds[:, 1], preds[:, 2]
        bg_true, fe_true, bm_true = targets[:, 0], targets[:, 1], targets[:, 2]
        
        mse_bg = F.mse_loss(bg_pred, bg_true)
        mse_fe = F.mse_loss(fe_pred, fe_true)
        mse_bm = F.mse_loss(bm_pred, bm_true)
        
        return mse_bg + mse_fe + mse_bm

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y
        if self.scaler is not None:
            targets = self.scaler.transform(targets)
        loss = self.custom_loss(preds, targets)
        self.log('train_loss', loss, batch_size=batch.num_graphs)
        self.train_step_outputs.append(loss)
        return loss
        
    def on_train_epoch_start(self):
        self.train_step_outputs = []
        
    def on_train_epoch_end(self):
        if self.train_step_outputs:
            avg_loss = torch.stack(self.train_step_outputs).mean()
            self.train_epoch_losses.append(avg_loss.item())

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y
        if self.scaler is not None:
            targets = self.scaler.transform(targets)
        loss = self.custom_loss(preds, targets)
        self.log('val_loss', loss, batch_size=batch.num_graphs, prog_bar=True)
        self.val_step_outputs.append(loss)
        return loss
        
    def on_validation_epoch_start(self):
        self.val_step_outputs = []
        
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and self.val_step_outputs:
            avg_loss = torch.stack(self.val_step_outputs).mean()
            self.val_epoch_losses.append(avg_loss.item())
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
