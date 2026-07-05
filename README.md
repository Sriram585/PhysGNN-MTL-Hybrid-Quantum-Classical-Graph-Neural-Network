# PhysGNN-MTL: Hybrid Quantum-Classical Graph Neural Network

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-812067)
![PyTorch Lightning](https://img.shields.io/badge/Lightning-⚡-792ee5)

**PhysGNN-MTL** is an advanced hybrid Quantum-Classical Graph Neural Network designed for precise, multi-task prediction of material properties directly from crystal structures.

By combining the spatial message-passing capabilities of classical Graph Neural Networks (GNNs) with the high-dimensional feature extraction capabilities of Variational Quantum Circuits (VQCs), this model acts as a powerful tool for materials discovery.

## 🌟 Key Features

* **Hybrid Architecture**: Leverages PyTorch Geometric for classical crystal structure message passing and PennyLane for parameterized quantum circuits.
* **Residual Quantum Bypass**: Introduces a crucial residual concatenation step that combines classical GNN pooled features with the quantum circuit's output. This elegantly solves the vanishing gradient/barren plateau problem common in quantum machine learning!
* **Multi-Task Learning (MTL)**: Predicts multiple properties simultaneously (e.g., Band Gap and Formation Energy), using shared representation layers.
* **Automated Data Fetching**: Seamlessly integrates with the [Materials Project API](https://next-gen.materialsproject.org/api) (`mp-api`) to dynamically query, fetch, and construct PyG crystal graphs on the fly.
* **Class Imbalance Handling**: Automatically utilizes a `WeightedRandomSampler` to handle data skews (e.g., heavy bias towards metallic zero band-gap materials).

## 🧠 Model Architecture

1. **Classical GNN**: A 3-layer `TransformerConv` network with `BatchNorm` extracts structural representations from node features (atomic numbers, electronegativity, radii) and edge features (interatomic displacement vectors).
2. **Quantum Projection**: The pooled classical graph features are projected into a 4-dimensional latent space and encoded into 4 qubits via `AngleEmbedding`.
3. **Variational Quantum Circuit**: A `StronglyEntanglingLayers` circuit processes the embedded state to extract complex quantum correlations.
4. **Residual Fusion & MLP Heads**: The outputs of the quantum circuit are concatenated *with* the classical graph features. Two independent Multi-Layer Perceptrons (MLPs) then predict the target physical properties.

## 🛠 Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd PhysGNN-MTL-Hybrid-Quantum-Classical-Graph-Neural-Network

# It is recommended to use a virtual environment
pip install torch torch-geometric pytorch-lightning
pip install pennylane pennylane-lightning[gpu] 
pip install mp-api matplotlib scikit-learn python-dotenv
```

## 🚀 Usage

### 1. API Key Setup
To fetch real materials data, you need an API key from the Materials Project. Create a `.env` file in the root directory:

```env
MP_API_KEY="your_api_key_here"
```

### 2. Training the Model
Run the main script to start fetching data and training the hybrid network.

```bash
python main.py
```

*By default, the script will fetch 1500 samples, train for 100 epochs using Early Stopping, and save checkpoints to the `checkpoints/` directory.*

### 3. Interactive Notebook
For an interactive, step-by-step exploration of the pipeline, you can run the provided Jupyter notebook:
- `Phys-GNN-Quantum.ipynb`: Contains the complete data, model, and training loop in a single GPU-accelerated notebook environment.

## 📁 Project Structure

```text
├── main.py                     # Entry point for the PyTorch Lightning training pipeline
├── Phys-GNN-Quantum.ipynb      # Complete interactive notebook implementation
├── README.md                   # Project documentation
├── checkpoints/                # Auto-generated directory for best model weights
├── outputs/                    # Auto-generated plots (Parity plots, Training curves)
└── src/                        
    ├── data/                   
    │   └── dataset.py          # MP-API data fetching and PyG CrystalDataset construction
    ├── models/                 
    │   └── physgnn.py          # Hybrid Quantum-Classical MTL architecture definition
    └── utils/                  
        ├── evaluation.py       # Model evaluation and matplotlib plotting utilities
        └── scaler.py           # TargetScaler for standardizing physical property distributions
```

## 📊 Outputs

Upon completion of the evaluation phase, the script evaluates the best model checkpoint on the test set and generates high-quality plots in the `outputs/` directory:

1. **`parity_plots.png`**: Visualizes `True` vs `Predicted` values for the target properties, displaying $R^2$ and MAE metrics.
2. **`training_curves.png`**: Displays the historical loss curves across training and validation epochs.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.