import torchvision
import torch


def data_loader(path, batch_size=10, mean=None, std=None):
    """
    A function that creates a dataloader for the root directory of a dataset split such as train, test, or validation.
    This function also applies some transforms on the data such as normalizing the data and converting the image to a tensor.

    Arguments:
        path {str} -- path to root folder of a subset of dataset ie. ../datasets/cifar10/train

    Keyword Arguments:
        batch_size {int} -- the size of each batch in the train loader (default: {10})
        mean {tuple} -- tuple containing the mean of each colour channel in the images (default: {None})
        std {tuple} -- tuple containing the standard devication of each colour channel in the dataset (default: {None})

    Returns:
        torch.utils.data.DataLoader -- an innitialised dataloader object containing len(dataset)/batch_size batches
    """

    transforms = [torchvision.transforms.RandomResizedCrop(32),
                  torchvision.transforms.RandomRotation(10),
                  torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.RandomVerticalFlip(),
                  torchvision.transforms.RandomPerspective(),
                  torchvision.transforms.ToTensor()
                  ]
    if mean is not None and std is not None:
        transforms.append(torchvision.transforms.Normalize(mean, std))

    transform = torchvision.transforms.Compose(transforms)

    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=transform)

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
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
