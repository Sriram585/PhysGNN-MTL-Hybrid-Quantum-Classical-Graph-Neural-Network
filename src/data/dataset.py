import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import WeightedRandomSampler, Dataset

class CrystalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def get_dataloader(data_list, batch_size=32, is_train=True):
    dataset = CrystalDataset(data_list)
    if is_train:
        # Class Imbalance Fix: is_metal = (band_gap == 0.0)
        labels = [1 if data.y[0, 0].item() == 0.0 else 0 for data in data_list]
        num_metals = sum(labels)
        num_non_metals = len(labels) - num_metals
        
        weight_metal = 1.0 / num_metals if num_metals > 0 else 1.0
        weight_non_metal = 1.0 / num_non_metals if num_non_metals > 0 else 1.0
        
        weights = [weight_metal if l == 1 else weight_non_metal for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return PyGDataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

def fetch_materials_project_data(api_key, num_samples=1500, cutoff=4.0):
    """
    Fetches real crystal structures from the Materials Project.
    Requires `mp-api` package: pip install mp-api
    """
    from mp_api.client import MPRester
    from tqdm import tqdm
    import warnings
    import os
    
    # Define cache path
    os.makedirs("data", exist_ok=True)
    cache_path = f"data/processed_dataset_{num_samples}.pt"
    
    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading {num_samples} cached materials from {cache_path}...")
        data_list = torch.load(cache_path, weights_only=False)
        # Fix NaNs and clip database outliers in cached data
        for data in data_list:
            data.x = torch.nan_to_num(data.x, nan=0.0)
            data.y = torch.nan_to_num(data.y, nan=0.0)
            # Clamp targets to reasonable physical bounds
            data.y[0, 0] = torch.clamp(data.y[0, 0], min=0.0, max=15.0)      # Band Gap
            data.y[0, 1] = torch.clamp(data.y[0, 1], min=-10.0, max=10.0)    # Formation Energy
        return data_list
        
    print(f"Querying Materials Project for up to {num_samples} materials...")
    
    # Suppress pymatgen warnings for cleaner output
    warnings.filterwarnings("ignore", module="pymatgen")
    
    with MPRester(api_key) as mpr:
        import math
        chunk_size = min(num_samples, 1000)
        num_chunks = math.ceil(num_samples / chunk_size)
        
        docs = mpr.materials.summary.search(
            fields=["structure", "band_gap", "formation_energy_per_atom", "bulk_modulus"],
            num_chunks=num_chunks,
            chunk_size=chunk_size
        )
        
    data_list = []
    
    # Take only the requested number of samples
    docs = docs[:num_samples]
    
    print("Converting structures to PyTorch Geometric graphs...")
    for doc in tqdm(docs):
        structure = doc.structure
        
        # Targets
        band_gap = doc.band_gap or 0.0
        form_energy = doc.formation_energy_per_atom or 0.0
        
        # Bulk modulus extraction (kept for potential future use)
        bulk_mod_val = getattr(doc, "bulk_modulus", None)
        if isinstance(bulk_mod_val, dict):
            bulk_mod = float(bulk_mod_val.get('vrh', 0.0))
        elif hasattr(bulk_mod_val, 'vrh'):
            bulk_mod = float(bulk_mod_val.vrh or 0.0)
        elif bulk_mod_val is not None:
            try:
                bulk_mod = float(bulk_mod_val)
            except (ValueError, TypeError):
                bulk_mod = 0.0
        else:
            bulk_mod = 0.0
            
        y = torch.tensor([[band_gap, form_energy]], dtype=torch.float)
        y = torch.nan_to_num(y, nan=0.0)
        y[0, 0] = torch.clamp(y[0, 0], min=0.0, max=15.0)
        y[0, 1] = torch.clamp(y[0, 1], min=-10.0, max=10.0)
        
        # Nodes
        node_features = []
        for site in structure:
            element = site.specie
            
            # Extract exactly 10 properties to match node_dim=10
            feat = [
                element.Z,
                element.group,
                element.row,
                element.atomic_mass,
                element.atomic_radius,
                element.X,  # electronegativity
                element.melting_point,
                element.boiling_point,
                element.density_of_solid,
                element.mendeleev_no
            ]
            # Replace any None with 0.0 safely
            feat = [f if f is not None else 0.0 for f in feat]
            node_features.append(feat)
            
        x = torch.tensor(node_features, dtype=torch.float)
        x = torch.nan_to_num(x, nan=0.0)
        
        # Edges
        # get_all_neighbors returns a list of lists of Neighbor objects within cutoff radius
        all_neighbors = structure.get_all_neighbors(r=cutoff)
        
        edge_index = []
        edge_attr = []
        
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                # Vector points from site i to neighbor j
                disp_vector = neighbor.coords - structure[i].coords
                
                edge_index.append([i, j])
                edge_attr.append(disp_vector.tolist())
                
        if len(edge_index) == 0:
            # If a structure is too sparse for the cutoff, skip it to avoid PyG crashing
            continue
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)
        
    print(f"Saving {len(data_list)} processed graphs to {cache_path}...")
    torch.save(data_list, cache_path)
    
    return data_list
