from torchvision import datasets, transforms
import torch


class MNISTData:

    def __init__(self, batch_size = 50, num_workers = 3):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('../data', train = True, download = True, transform = transform)
        self.data_loader = torch.utils.data.DataLoader(dataset1, batch_size = batch_size,  num_workers = num_workers)

    def load_batch(self):
        """
        Returns:
        data => torch tensor of size (batch_size, 28, 28)
        target => torch tensor of size (batch_size)
        """
        data, target = next(iter(self.data_loader))
        data = torch.squeeze(data)
        return data, target

    def load_flattened_batch(self):
        """
        Returns:
        data => numpy array of size (784, batch_size)
        target => numpy array of size (1, batch_size)
        """
        data, target = self.load_batch()
        flattened_input = data.numpy().reshape((-1, data.shape[0]))
        flattened_labels = target.numpy().reshape((1, -1))
        return flattened_input, flattened_labels
