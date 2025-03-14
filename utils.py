import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 获取数据
def get_data_loaders(num_clients, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 分割数据
    client_data = torch.utils.data.random_split(train_set, [len(train_set) // num_clients] * num_clients)
    data_loaders = [torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=True) for cd in client_data]
    return data_loaders, torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 后门攻击：篡改数据
def poison_data(data_loader, test=False):
    backdoor_label = 0  # 目标类别
    for images, labels in data_loader:
        images[:, :, 0, 0] = 1.0  # 在左上角加白色像素
        labels[:] = backdoor_label
    return data_loader

# Coordinate Trimmed Mean 防御
def coordinate_trimmed_mean(param_list, trim_ratio=0.2):
    stacked_params = torch.stack([torch.flatten(torch.tensor(list(p.values()))) for p in param_list])
    sorted_params, _ = torch.sort(stacked_params, dim=0)
    trim_n = int(trim_ratio * len(param_list))
    return {k: v for k, v in zip(param_list[0].keys(), torch.mean(sorted_params[trim_n:-trim_n], dim=0).split(128))}
