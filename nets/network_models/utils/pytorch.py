import torchvision
import torch


def data_loader(path, batch_size=10):
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         ])

    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=transform)

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )
    return dataset_loader


def get_normalized_dataset_vals(data_loader):
    '''
        method created using this resource
        https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    '''
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        """
        always will be data[0] as data[1] is labels
        loop through data[0][i] - this is each image in the batch
        data[0][i][j] is the channel
        """
        batch_samples = data[0].size(0)
        data = data[0].view(batch_samples, data[0].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

    mean /= len(data_loader.dataset)
    print(mean)
    std /= len(data_loader.dataset)
    print(std)
