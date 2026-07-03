import torch
import numpy as np
from torch_geometric.data import Data, DataLoader as PyGDataLoader
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

def fetch_materials_project_data(api_key, num_samples=100):
    """
    Fetches real crystal structures from the Materials Project.
    Requires `mp-api` package: pip install mp-api
    """
    # from mp_api.client import MPRester
    
    # with MPRester(api_key) as mpr:
    #     docs = mpr.materials.summary.search(
    #         band_gap=(0, 10),
    #         fields=["material_id", "structure", "band_gap", "formation_energy_per_atom", "bulk_modulus"]
    #     )
    #     
    #     # TODO: Convert pymatgen structures (`doc.structure`) to PyTorch Geometric graphs
    #     # (e.g. using distance cutoffs to build edges, atomic numbers for node features)
    
    raise NotImplementedError("Real MP data fetching and graph conversion requires implementation.")

def generate_dummy_data(num_samples=100):
    """Generates synthetic Materials Project-like data for testing the pipeline."""
    data_list = []
    for _ in range(num_samples):
        num_nodes = np.random.randint(5, 20)
        num_edges = np.random.randint(10, 40)
        
        x = torch.randn(num_nodes, 10) 
        edge_index = torch.randint(0, num_nodes, (2, num_edges)) 
        edge_attr = torch.randn(num_edges, 3)
        
        is_metal = np.random.rand() > 0.5
        band_gap = 0.0 if is_metal else np.random.uniform(0.1, 5.0)
        form_energy = np.random.uniform(-5.0, 1.0)
        bulk_mod = np.random.uniform(20.0, 300.0)
        
        y = torch.tensor([[band_gap, form_energy, bulk_mod]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)
    return data_list
