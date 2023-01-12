from torchvision import datasets, transforms
import torch


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train = True, download = True, transform = transform)
data_loader = torch.utils.data.DataLoader(dataset1, batch_size = 50,  num_workers = 3)


import sys
sys.path.append("build/")

from CPPDataHandler import ReadData
cnt = 0
for batch_idx, (data, target) in enumerate(data_loader):
    if cnt > 1:
        break
    flattened_data = data.reshape((50, 784))
    print(flattened_data.shape, target.shape)
    cnt += 1
    DataHandlerObject = ReadData(flattened_data, target)





