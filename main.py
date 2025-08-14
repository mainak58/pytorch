import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset: folder structure
# dataset/
#    charts/
#    non_charts/
dataset = datasets.ImageFolder(root='dataset', transform=transform)

# DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Check classes
print(dataset.classes)  # ['charts', 'non_charts']

# Example iteration
for imgs, labels in dataloader:
    print(imgs.shape, labels)
    break
