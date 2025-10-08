from torch.utils.data import DataLoader




dataloader = DataLoader()


for i in range(max_steps):
    # Optimization steps
    data = data.to(device=device, dtype=dtype)
