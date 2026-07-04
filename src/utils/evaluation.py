import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_and_plot(model, dataloader, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch)
            if hasattr(model, 'scaler') and model.scaler is not None:
                preds = model.scaler.inverse_transform(preds)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch.y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    properties = ['Band Gap', 'Formation Energy', 'Bulk Modulus']
    
    # --- Parity Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i in range(3):
        true_vals = all_trues[:, i]
        pred_vals = all_preds[:, i]
        
        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
        
        ax = axes[i]
        ax.scatter(true_vals, pred_vals, alpha=0.6, edgecolors='k', color='dodgerblue')
        
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        
        ax.set_title(f'{properties[i]} Parity Plot')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        
        textstr = f'MAE = {mae:.4f}\\n$R^2$ = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        ax.legend(loc='lower right')
                
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parity_plots.png'), dpi=300)
    plt.close()
    
    # --- Training Curves ---
    if len(model.train_epoch_losses) > 0 and len(model.val_epoch_losses) > 0:
        plt.figure(figsize=(8, 6))
        epochs = range(1, len(model.train_epoch_losses) + 1)
        plt.plot(epochs, model.train_epoch_losses, label='Train Loss', marker='o', color='blue')
        plt.plot(epochs, model.val_epoch_losses, label='Validation Loss', marker='x', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
    print(f"Evaluation complete. Plots saved to '{output_dir}/' directory.")
