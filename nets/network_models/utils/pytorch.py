import torchvision
import torch


def data_loader(path, batch_size=10, mean=None, std=None):
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms = [torchvision.transforms.ToTensor()]
    if mean is not None and std is not None:
        transforms.append(torchvision.transforms.Normalize(mean, std))

    transform = torchvision.transforms.Compose(transforms)

    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=transform)

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )
    return dataset_loader


def get_normalized_dataset_vals(path):
    '''
        method created using this resource
        https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    '''
    mean = 0.
    std = 0.
    d_loader = data_loader(path)
    for data in d_loader:
        """
        always will be data[0] as data[1] is labels
        """
        batch_samples = data[0].size(0)
        data = data[0].view(batch_samples, data[0].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

    mean /= len(d_loader.dataset)
    std /= len(d_loader.dataset)

    return tuple(mean.tolist()), tuple(std.tolist())
