import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
import pytorch_lightning as pl
import pennylane as qml

# Configure the PennyLane Quantum Device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class PhysGNN_MTL(pl.LightningModule):
    def __init__(self, node_dim, edge_dim, n_qubits=4, q_depth=2, lr=5e-3, scaler=None):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler'])
        self.lr = lr
        self.scaler = scaler
        
        # --- Classical Graph Module ---
        self.conv1 = TransformerConv(node_dim, 64, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = TransformerConv(64, 128, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = TransformerConv(128, 128, edge_dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(128)
        
        # --- Projection to quantum input ---
        self.fc_to_quantum = nn.Linear(128, n_qubits)
        
        # --- Quantum Circuit Module ---
        weight_shapes = {"weights": (q_depth, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # --- Multi-Task Heads ---
        # Takes BOTH classical pooled features (128) AND quantum features (n_qubits)
        combined_dim = 128 + n_qubits
        
        self.head_band_gap = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.head_form_energy = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Tracking losses for analytical plotting
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Classical GNN
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        
        # Global Pooling → 128-dim classical features
        x_pool = global_mean_pool(x, batch)
        
        # Quantum branch
        x_proj = self.fc_to_quantum(x_pool)
        x_q_in = torch.sigmoid(x_proj) * torch.pi
        q_out = self.qlayer(x_q_in)
        
        # RESIDUAL: Concatenate classical features WITH quantum features
        combined = torch.cat([x_pool, q_out], dim=1)
        
        # Regression Heads
        bg = self.head_band_gap(combined)
        fe = self.head_form_energy(combined)
        
        return torch.cat([bg, fe], dim=1)

    def custom_loss(self, preds, targets):
        bg_pred, fe_pred = preds[:, 0], preds[:, 1]
        bg_true, fe_true = targets[:, 0], targets[:, 1]
        
        loss_bg = F.huber_loss(bg_pred, bg_true, delta=1.0)
        loss_fe = F.huber_loss(fe_pred, fe_true, delta=1.0)
        
        return loss_bg + loss_fe

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y[:, :2]
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
        targets = batch.y[:, :2]
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
