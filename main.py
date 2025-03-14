import torch
import numpy as np
from federated_learning import FederatedLearning
from model import SimpleNN
from utils import get_data_loaders, poison_data

# 配置参数
num_clients = 10      # 客户端数量
num_malicious = 3     # 恶意客户端数量
num_rounds = 50       # 训练轮数
batch_size = 64       # 批次大小
learning_rate = 0.01  # 学习率
backdoor_label = 0    # 后门攻击目标标签
trim_ratio = 0.2      # Coordinate Trimmed Mean 截断比例

# 初始化 FL 训练
fl = FederatedLearning(
    num_clients=num_clients, num_malicious=num_malicious,
    model=SimpleNN(), lr=learning_rate, batch_size=batch_size
)

# 训练前后门测试
fl.test_global_model()

# 训练过程
for round in range(num_rounds):
    print(f"\n--- Round {round + 1} ---")
    fl.train_one_round(backdoor=True)  # 进行后门攻击
    fl.aggregate_parameters(defense='CTM', trim_ratio=trim_ratio)  # 采用 Coordinate Trimmed Mean 进行防御
    fl.test_global_model()

# 训练完成后门攻击测试
print("\n--- After Training (Testing Backdoor Attack) ---")
fl.test_global_model(poisoned=True)
