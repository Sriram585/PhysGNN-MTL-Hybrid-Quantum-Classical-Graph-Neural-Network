import torch

class TargetScaler:
    """Standardizes target properties to have mean=0 and std=1 to fix MTL scale imbalance."""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, dataloader):
        all_y = []
        for batch in dataloader:
            all_y.append(batch.y)
        all_y = torch.cat(all_y, dim=0)
        self.mean = all_y.mean(dim=0, keepdim=True)
        self.std = all_y.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
    def transform(self, y):
        if self.mean is None or self.std is None:
            return y
        return (y - self.mean.to(y.device)) / self.std.to(y.device)
        
    def inverse_transform(self, y_scaled):
        if self.mean is None or self.std is None:
            return y_scaled
        return (y_scaled * self.std.to(y_scaled.device)) + self.mean.to(y_scaled.device)
