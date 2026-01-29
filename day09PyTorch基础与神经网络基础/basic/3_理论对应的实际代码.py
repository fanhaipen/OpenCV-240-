# 理论对应的实际代码
import torch
import torch.nn as nn


# 1. 定义神经网络（对应理论中的层次结构）
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 权重和偏置在这里自动创建
        self.layer1 = nn.Linear(2, 3)  # 2输入→3个神经元
        self.layer2 = nn.Linear(3, 1)  # 3神经元→1输出
        self.activation = nn.ReLU()  # 激活函数

    # 2. 前向传播（数据流动路径）
    def forward(self, x):
        print(f"输入: {x}")  # 原始数据

        # 第一层处理
        x = self.layer1(x)  # 自动计算：输入×权重 + 偏置
        print(f"第一层后: {x}")

        x = self.activation(x)  # 激活函数决定哪些神经元"触发"
        print(f"激活后: {x}")

        # 第二层处理
        x = self.layer2(x)
        print(f"最终输出: {x}")

        return x


# 创建网络实例

layer = nn.Linear(2, 3)  # 输入2，输出3

print("=== 权重和偏置的验证 ===")
print(f"权重形状: {layer.weight.shape}")  # 应该是(3, 2)
print(f"偏置形状: {layer.bias.shape}")    # 应该是(3,)

print(f"\n权重值:")
print(layer.weight.data)

print(f"\n偏置值:")
print(layer.bias.data)


model = SimpleNet()

# 测试前向传播
sample_input = torch.tensor([[1.0, 2.0]])  # 样本数据
print("=== 前向传播过程 ===")
output = model(sample_input)  # 这里隐式调用了forward方法！

print(f"\n网络参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")