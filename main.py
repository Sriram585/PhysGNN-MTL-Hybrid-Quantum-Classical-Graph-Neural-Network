import os
import shutil
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from src.data.dataset import get_dataloader, fetch_materials_project_data
from src.models.physgnn import PhysGNN_MTL
from src.utils.evaluation import evaluate_and_plot
from src.utils.scaler import TargetScaler

def main():
    # --- Configuration ---
    epochs = 100
    batch_size = 32
    num_samples = 1500
    api_key = os.getenv("MP_API_KEY")
    
    # Fix seeds for reproducibility
    pl.seed_everything(42)
    
    if not api_key:
        raise ValueError("Materials Project API Key is required. Set MP_API_KEY in .env or pass your key.")
        
    print(f"Connecting to Materials Project using API Key: {api_key[:4]}...")
    dataset = fetch_materials_project_data(api_key, num_samples)
        
    # --- Class Imbalance Check ---
    metals = sum(1 for data in dataset if data.y[0, 0].item() == 0.0)
    non_metals = len(dataset) - metals
    print(f"[DATASET STATS] Total: {len(dataset)} | Metals: {metals} ({metals/len(dataset)*100:.1f}%) | Non-Metals: {non_metals} ({non_metals/len(dataset)*100:.1f}%)")
    
    # Simple split
    num_train = int(0.7 * len(dataset))
    num_val = int(0.15 * len(dataset))
    
    train_dataset = dataset[:num_train]
    val_dataset = dataset[num_train:num_train+num_val]
    test_dataset = dataset[num_train+num_val:]
    
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, is_train=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, is_train=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, is_train=False)
    
    # Fit scaler to training dataset
    scaler = TargetScaler()
    scaler.fit(train_loader)
    
    # Initialize model
    print("Initializing PhysGNN-MTL model...")
    model = PhysGNN_MTL(node_dim=10, edge_dim=3, n_qubits=4, scaler=scaler)
    
    # Delete old checkpoints to avoid loading stale weights
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    os.makedirs('checkpoints', exist_ok=True)
    
    # Checkpointing: best val_loss and latest epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True 
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=25,
        mode='min'
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=5,
        logger=False
    )
    
    print("Starting training pipeline...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save training history BEFORE loading checkpoint (checkpoint creates a new object)
    saved_train_losses = model.train_epoch_losses.copy()
    saved_val_losses = model.val_epoch_losses.copy()
    
    print("Loading best model for testing and plotting...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = PhysGNN_MTL.load_from_checkpoint(best_model_path, scaler=scaler)
    
    # Restore training history onto the loaded model
    model.train_epoch_losses = saved_train_losses
    model.val_epoch_losses = saved_val_losses
    
    evaluate_and_plot(model, test_loader)
    print("All tasks completed successfully!")

if __name__ == '__main__':
    main()
