import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from utils import get_data_loaders, poison_data, coordinate_trimmed_mean
from model import SimpleNN

class FederatedLearning:
    def __init__(self, num_clients, num_malicious, model, lr, batch_size):
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取数据
        self.data_loaders = get_data_loaders(num_clients, batch_size)
        self.malicious_clients = np.random.choice(num_clients, num_malicious, replace=False)  # 随机选择恶意客户端
        self.global_parameters = copy.deepcopy(self.model.state_dict())  # 初始化全局参数
    
    def train_one_round(self, backdoor=False):
        local_parameters = []
        for client_id in range(self.num_clients):
            model = SimpleNN().to(self.device)
            model.load_state_dict(copy.deepcopy(self.global_parameters))
            optimizer = optim.SGD(model.parameters(), lr=self.lr)

            # 获取数据
            if client_id in self.malicious_clients and backdoor:
                train_loader = poison_data(self.data_loaders[client_id])  # 进行后门攻击
            else:
                train_loader = self.data_loaders[client_id]

            # 训练
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(images), labels)
                loss.backward()
                optimizer.step()

            local_parameters.append(copy.deepcopy(model.state_dict()))

        self.local_parameters = local_parameters

    def aggregate_parameters(self, defense='CTM', trim_ratio=0.2):
        if defense == 'CTM':  # 使用 Coordinate Trimmed Mean 防御
            self.global_parameters = coordinate_trimmed_mean(self.local_parameters, trim_ratio)
        else:
            self.global_parameters = copy.deepcopy(self.local_parameters[0])

    def test_global_model(self, poisoned=False):
        model = SimpleNN().to(self.device)
        model.load_state_dict(copy.deepcopy(self.global_parameters))
        model.eval()

        _, test_loader = get_data_loaders(1, self.batch_size)
        correct, total = 0, 0

        for images, labels in test_loader:
            if poisoned:
                images, labels = poison_data([(images, labels)], test=True)[0]
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Test Accuracy ({'Poisoned' if poisoned else 'Clean'}): {acc:.2f}%")
