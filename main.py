import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.dataset import generate_dummy_data, get_dataloader
from src.models.physgnn import PhysGNN_MTL
from src.utils.evaluation import evaluate_and_plot

def main():
    parser = argparse.ArgumentParser(description="PhysGNN-MTL Training Pipeline")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_samples', type=int, default=300, help="Number of synthetic samples to generate")
    parser.add_argument('--api_key', type=str, default=None, help="Materials Project API Key (or set MP_API_KEY in .env)")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Skipping .env file loading.")
        
    # Use argument if provided, otherwise fallback to .env variable
    api_key = args.api_key or os.getenv("MP_API_KEY")
    
    # Fix seeds for reproducibility
    pl.seed_everything(42)
    
    if api_key:
        print(f"Connecting to Materials Project using API Key: {api_key[:4]}...")
        # dataset = fetch_materials_project_data(api_key, args.num_samples)
        print("Note: Materials Project fetching is currently a stub. Falling back to dummy data.")
        dataset = generate_dummy_data(args.num_samples)
    else:
        print(f"Generating {args.num_samples} synthetic Materials Project data points...")
        dataset = generate_dummy_data(args.num_samples) 
    
    # Simple split
    num_train = int(0.7 * args.num_samples)
    num_val = int(0.15 * args.num_samples)
    
    train_data = dataset[:num_train]
    val_data = dataset[num_train:num_train+num_val]
    test_data = dataset[num_train+num_val:]
    
    train_loader = get_dataloader(train_data, batch_size=args.batch_size, is_train=True)
    val_loader = get_dataloader(val_data, batch_size=args.batch_size, is_train=False)
    test_loader = get_dataloader(test_data, batch_size=args.batch_size, is_train=False)
    
    # Initialize model
    print("Initializing PhysGNN-MTL model...")
    model = PhysGNN_MTL(node_dim=10, edge_dim=3)
    
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
        patience=20,
        mode='min'
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='cpu',
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    
    print("Starting training pipeline...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Loading best model for testing and plotting...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = PhysGNN_MTL.load_from_checkpoint(best_model_path)
    
    evaluate_and_plot(model, test_loader)
    print("All tasks completed successfully!")

if __name__ == '__main__':
    main()
