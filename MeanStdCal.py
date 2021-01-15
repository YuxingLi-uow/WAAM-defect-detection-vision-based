import torch
from torchvision import datasets, transforms
import os


def MeanStdCal(data_dir, data_set):

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transforms.ToTensor())
        for x in [data_set]
    }

    loader = torch.utils.data.DataLoader(
        image_datasets[data_set],
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print('mean: {}'.format(mean))
    print('std: {}'.format(std))

    return mean, std


if __name__ == '__main__':

    data_dir = 'DatasetRGBraw'
    dataset_cal = 'test'

    mean, std = MeanStdCal(data_dir, dataset_cal)

